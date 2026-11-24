"""Serve the StairLab GPU workspace through an SSH tunnel."""

import argparse
import base64
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from stairlab.recognition.depthout_refine import DepthOutFragments
from stairlab.recognition.pipeline import run_case
from stairlab.recognition.geometry import square
from stairlab.depth.predict_relative_depth import colorize
from stairlab.recognition.dense_head import DenseStairHead
from stairlab.recognition.yolo_fragments import YoloFragments


LIMIT = 12 * 1024 * 1024
EXAMPLES = {'front': 'color_110.png', 'dark': 'color_0.png', 'angled': 'color_74.png'}
ASSETS = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'),
          '/app.js': ('app.js', 'text/javascript')}


def encode(data):
    return base64.b64encode(data).decode('ascii')


def image_bytes(image, kind='PNG'):
    buffer = io.BytesIO()
    image.save(buffer, format=kind)
    return buffer.getvalue()


def decode(data):
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in {'JPEG', 'PNG', 'WEBP'}:
            raise ValueError('请选择 JPEG、PNG 或 WebP 图片。')
        if source.width * source.height > 25_000_000 or min(source.size) < 24:
            raise ValueError('图像尺寸超出范围，至少 24 × 24 像素。')
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    return image


class Pipeline:
    def __init__(self, args):
        self.device = args.device
        weights = torch.load(args.yolo, map_location='cpu', weights_only=True)
        self.yolo = YoloFragments('yolo11n-pose.yaml', surface_aux=weights.get('surface_aux', False)).to(self.device).eval()
        self.yolo.load_state_dict(weights['model'])
        self.surfaces = DepthOutFragments(args.depthout, args.depthout_weights, self.device)
        self.dense = DenseStairHead().to(self.device).eval()
        self.dense.load_state_dict(torch.load(args.dense, map_location='cpu', weights_only=True))
        self.processor = AutoImageProcessor.from_pretrained(args.depth_model, local_files_only=True)
        self.depth = AutoModelForDepthEstimation.from_pretrained(args.depth_model, local_files_only=True)
        ckpt = torch.load(args.decoder, map_location='cpu', weights_only=True)
        expected = {name for name in self.depth.state_dict() if not name.startswith('backbone.')}
        if ckpt.get('version') != '4.0' or set(ckpt['decoder']) != expected:
            raise ValueError('Expected a StairLab 4.0 decoder checkpoint')
        self.depth.load_state_dict(ckpt['decoder'], strict=False)
        self.depth.to(self.device).eval()
        self.size = ckpt['size']

    @torch.inference_mode()
    def run(self, image):
        start = perf_counter()
        prepared, offset, scale = square(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        detection = run_case(prepared, None, self.yolo, self.surfaces, self.dense, self.device, .30,
                             include_geometry=True)
        groups = [(np.asarray(lines) / scale - offset).round(1).tolist()
                  for lines in detection['edges_3_9']]
        if len(groups) != detection['count_3_9_refine']:
            raise RuntimeError('Edge groups and count disagree')
        groups.sort(key=lambda group: float(np.asarray(group)[..., 1].mean()), reverse=True)

        side = max(image.size)
        left, top = (side - image.width) // 2, (side - image.height) // 2
        padded = Image.new('RGB', (side, side))
        padded.paste(image, (left, top))
        pixels = self.processor(images=padded, return_tensors='pt', keep_aspect_ratio=False,
                                size={'height': self.size, 'width': self.size})['pixel_values'].to(self.device)
        depth = self.depth(pixel_values=pixels).predicted_depth
        depth = torch.nn.functional.interpolate(depth[:, None], (side, side), mode='bilinear', align_corners=False)[0, 0]
        depth = depth[top:top + image.height, left:left + image.width].float().cpu().numpy()
        if not np.isfinite(depth).all():
            raise RuntimeError('Nonfinite relative depth')
        heat, limits = colorize(depth)
        raw = io.BytesIO()
        np.save(raw, depth.astype('<f4', copy=False), allow_pickle=False)
        return {'width': image.width, 'height': image.height, 'count': len(groups), 'groups': groups,
                'rgb': encode(image_bytes(image, 'JPEG')), 'heat': encode(image_bytes(heat)),
                'depth_npy': encode(raw.getvalue()), 'display_range': limits,
                'depth_kind': 'relative_inverse_not_meters', 'seconds': round(perf_counter() - start, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('yolo', 'depthout', 'depthout-weights', 'dense', 'depth-model', 'decoder', 'examples'):
        parser.add_argument('--' + option, required=True, type=Path)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    app = Pipeline(args)
    busy = Lock()
    assets = Path(__file__).with_name('web_ui')

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, data, content_type='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', content_type + ('; charset=utf-8' if content_type.startswith('text/') else ''))
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def respond(self, code, data):
            self.send(code, json.dumps(data, ensure_ascii=False).encode('utf-8'))

        def do_GET(self):
            if self.path in ASSETS:
                name, kind = ASSETS[self.path]
                self.send(200, (assets / name).read_bytes(), kind)
            elif self.path == '/api/health':
                self.respond(200, {'ready': True, 'device': args.device})
            elif self.path.startswith('/api/example/') and self.path.rsplit('/', 1)[-1] in EXAMPLES:
                name = EXAMPLES[self.path.rsplit('/', 1)[-1]]
                self.send(200, (args.examples / name).read_bytes(), 'image/png')
            else:
                self.respond(404, {'error': '页面不存在。'})

        def do_POST(self):
            if self.path != '/api/analyze':
                return self.respond(404, {'error': '接口不存在。'})
            try:
                size = int(self.headers.get('Content-Length', 0))
            except ValueError:
                size = 0
            if not 0 < size <= LIMIT:
                return self.respond(413, {'error': '请选择小于 12 MB 的图片。'})
            if not busy.acquire(blocking=False):
                return self.respond(503, {'error': 'GPU 正在处理另一张图，请稍后重试。'})
            try:
                image = decode(self.rfile.read(size))
                result = app.run(image)
            except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError):
                return self.respond(400, {'error': '无法读取图像。请选择尺寸合适的 JPEG、PNG 或 WebP 图片。'})
            except Exception:
                logging.exception('StairLab inference failed')
                return self.respond(500, {'error': '推理失败，请查看服务器日志。'})
            finally:
                busy.release()
            self.respond(200, result)

    for name in EXAMPLES.values():
        if not (args.examples / name).is_file():
            raise FileNotFoundError(args.examples / name)
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'StairLab ready at http://127.0.0.1:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
