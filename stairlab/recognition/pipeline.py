import cv2
import numpy as np

from stairlab.recognition.depthout_stitch import predict_steps
from stairlab.recognition.full_flight_tracks import refine_flight
from stairlab.recognition.geometry import rgb_tensor, zoom
from stairlab.recognition.perspective_step_tracks import recover_perspective_tracks
from stairlab.recognition.yolo_fragments import SIZE, decode_batch


def propose_roi(mask):
    """Use the spatial extent of substantial stair-surface components."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask > 0))
    regions = stats[1:count]
    regions = regions[regions[:, 4] >= max(24, .0015 * mask.size)]
    if not len(regions):
        return None
    left, top = regions[:, :2].min(axis=0)
    right, bottom = (regions[:, :2] + regions[:, 2:4]).max(axis=0)
    pad = max(12, round(.08 * max(right-left, bottom-top)))
    box = [max(0, left-pad), max(0, top-pad), min(SIZE, right+pad), min(SIZE, bottom+pad)]
    if max(box[2]-box[0], box[3]-box[1]) >= .95 * SIZE:
        return None
    return list(map(int, box))



def groups(result):
    return [step['edge_segments'] for step in result['steps']]



def prep(data, mask, image, confidence):
    return groups(predict_steps(data, mask, recover_short_edges=True, fragment_confidence=confidence,
                                texture_image=image, split_dense_tracks=True))



def run_case(image, locked_roi, yolo, depth, dense, device, confidence, *, include_geometry=False):
    roi = locked_roi
    if roi is None:
        _, mask = depth.predict([image], confidence, with_masks=True)[0]
        roi = propose_roi(mask)
    if roi is None:
        crop, scale, offset = image, 1, [0, 0]
    else:
        crop, scale, offset = zoom(image, roi)
    frames = [image, crop]
    fragments = decode_batch(yolo(rgb_tensor(frames, device)), confidence)
    masks = [result[1] for result in depth.predict(frames, confidence, with_masks=True)]
    whole = prep(fragments[0], masks[0], image, confidence)
    output = {'roi': roi, 'base': 'whole', 'count_3_4': len(whole), 'count_3_8': len(whole),
              'count_3_9_refine': len(whole), 'count_3_9_add': len(whole), 'new_tracks': [], 'refinements': 0}
    if include_geometry:
        original = predict_steps(fragments[0], masks[0], fragment_confidence=confidence)
        output.update(edges_3_0=groups(original), edges_3_9=whole)
    if roi is None:
        return output
    local = [[(np.asarray(line)*scale + offset).tolist() for line in step]
             for step in prep(fragments[1], masks[1], crop, confidence)]
    mapped = np.concatenate([fragments[1][0]*scale + offset, fragments[0][0]])
    scores = np.concatenate([fragments[1][1], fragments[0][1]])
    heat = dense(rgb_tensor([crop], device)).sigmoid()[0, 0].cpu().numpy()
    heat = cv2.warpAffine(heat, np.array([[scale, 0, offset[0]], [0, scale, offset[1]]]), (SIZE, SIZE))
    added, diagnosis = recover_perspective_tracks(image, mapped, scores, local, roi, dense_response=heat, joint=True)
    if diagnosis.get('fit_error', float('inf')) > .40:
        return output
    exterior = [step for step in whole if all(max(y for _, y in line) < roi[1] or min(y for _, y in line) >= roi[3]
                                               for line in step)]
    anchored = local + exterior + added
    extended, new, refinements = refine_flight(image, mapped, scores, anchored, roi)
    output.update(base='crop', count_3_8=len(anchored), count_3_9_refine=len(extended)-len(new),
                  count_3_9_add=len(extended), new_tracks=new, refinements=len(refinements))
    if include_geometry:
        output['edges_3_9'] = extended[:len(anchored)]
    return output

