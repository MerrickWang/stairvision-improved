import unittest

import numpy as np

from stairlab.recognition.step_count import count_visible_steps
from stairlab.recognition.step_grouping import count_grouped_steps
from stairlab.recognition.depthout_stitch import dense_spacing, split_compound_edges, tread_texture_lines


class StepCountTests(unittest.TestCase):
    def test_split_reuses_only_supported_parallel_tracks(self):
        def line(y):
            return {'class': 0, 'confidence': .8, 'endpoints': [[100, y], [218, y]]}

        lines = [line(y) for y in (118, 128, 141, 153, 165)]
        fragments = np.array([[[x, y], [x + 14, y]] for y in (125, 131)
                              for x in range(100, 205, 16)], dtype=float)
        scores = np.full(len(fragments), .8)
        empty = np.array([], dtype=int)
        chains = [empty, np.arange(len(fragments)), empty, empty, empty]
        recovered, events, used = split_compound_edges(lines, chains, fragments, scores, (512, 512))
        self.assertEqual(len(events), 1)
        self.assertTrue(used.all())
        self.assertEqual(len(recovered), 6)
        self.assertEqual(count_grouped_steps([], recovered, (512, 512))['visible_step_count'], 6)
        chains[1] = np.arange(len(fragments) // 2)
        self.assertFalse(split_compound_edges(lines, chains, fragments, scores, (512, 512))[1])

    def test_split_pair_is_not_merged_as_two_nearby_copies(self):
        lines = [{'class': 0, 'confidence': .9, 'endpoints': [[70, y], [175, y]]}
                 for y in (110, 114)]
        self.assertEqual(count_grouped_steps([], lines, (512, 512))['visible_step_count'], 1)
        for line in lines:
            line['split_track_pair'] = 0
        self.assertEqual(count_grouped_steps([], lines, (512, 512))['visible_step_count'], 2)

    def test_dense_spacing_supports_gap_and_rejects_duplicate(self):
        def line(y):
            return {'endpoints': [[40, y], [220, y]]}

        known = [line(y) for y in (100, 115, 145, 160)]
        self.assertTrue(dense_spacing(line(130), known, (512, 512)))
        self.assertFalse(dense_spacing(line(116), known, (512, 512)))

    def test_texture_band_preserves_leading_edge_and_following_level(self):
        rng = np.random.default_rng(42)
        gray = np.full((512, 512), 105, dtype=np.uint8)
        gray[120:170] = rng.integers(40, 205, (50, 512), dtype=np.uint8)
        gray[250:] = 150
        image = np.repeat(gray[..., None], 3, axis=2)
        mask = np.ones(gray.shape, dtype=np.uint8)
        lines = [{'endpoints': [[20, y], [490, y]]} for y in (120, 170, 214, 250)]
        self.assertEqual(tread_texture_lines(lines, image, mask),
                         {1: 'trailing_texture_band', 2: 'weak_same_surface_seam'})

    def test_relaxed_length_applies_only_to_accepted_short_edges(self):
        short = {'class': 0, 'confidence': 0.95, 'endpoints': [[455, 258], [511, 271]]}
        marked = {**short, 'short_edge_recovery': 'local_level_spacing'}
        shape = (512, 512)
        self.assertEqual(count_grouped_steps([], [marked], shape)['visible_step_count'], 0)
        for line, expected in ((short, 0), (marked, 1)):
            with self.subTest(expected=expected):
                result = count_grouped_steps([], [line], shape, short_edge_minimum_length_ratio=.09)
                self.assertEqual(result['visible_step_count'], expected)

    shape = (128, 160)

    def mask(self, top, bottom):
        mask = np.zeros(self.shape, dtype=bool)
        for col in range(20, 140):
            slope = col // 10
            mask[top + slope:bottom + slope, col] = True
        return mask

    def line(self, y, confidence=0.9):
        return {'class': 0, 'confidence': confidence, 'endpoints': [[20, y + 2], [139, y + 13]]}

    def test_merges_two_close_edges_with_shared_surface(self):
        riser = self.mask(41, 55)
        surfaces = [(1, 0.91, riser), (1, 0.89, riser.copy()), (0, 0.8, self.mask(30, 42))]
        result = count_visible_steps(surfaces, [self.line(40), self.line(43)], self.shape)
        self.assertEqual(result['structured_convex_count'], 2)
        self.assertEqual(result['visible_step_count'], 1)
        self.assertEqual(result['steps'][0]['merged_edges'], 1)
        self.assertCountEqual(result['steps'][0]['evidence'], ['convex_edge', 'riser', 'tread'])

    def test_keeps_edges_without_any_surface_support(self):
        surfaces = [(0, 0.9, self.mask(30, 41))]
        result = count_visible_steps(surfaces, [self.line(40), self.line(70)], self.shape)
        self.assertEqual(result['visible_step_count'], 2)
        self.assertEqual(result['steps'][1]['confidence'], 'low')

    def test_recovers_only_supported_low_confidence_edge_in_a_gap(self):
        lines = [self.line(y) for y in (25, 45, 85, 105)]
        surfaces = [(1, 0.9, self.mask(65, 75)), (0, 0.9, self.mask(10, 16))]
        result = count_visible_steps(surfaces, lines + [self.line(65, 0.15), self.line(15, 0.15)], self.shape)
        self.assertEqual(result['visible_step_count'], 5)
        self.assertTrue(result['steps'][2]['recovered'])
        self.assertEqual(result['steps'][2]['evidence'], ['convex_edge', 'riser'])
        self.assertEqual(len(result['edge_stages']['deduplicated']), 4)

    def test_surfaces_do_not_add_a_step_without_a_low_edge(self):
        lines = [self.line(y) for y in (25, 45, 85, 105)]
        surfaces = [(1, 0.9, self.mask(65, 75)), (0, 0.9, self.mask(57, 66))]
        result = count_visible_steps(surfaces, lines, self.shape)
        self.assertEqual(result['visible_step_count'], 4)
        self.assertEqual(len(result['suspected_gaps']), 1)

    def test_merges_clear_duplicates_without_segmentation(self):
        result = count_visible_steps([], [self.line(40), self.line(42)], self.shape)
        self.assertEqual(result['visible_step_count'], 1)

    def test_snapping_changes_endpoints_without_changing_count(self):
        result = count_visible_steps([(1, 0.9, self.mask(42, 55))], [self.line(40)], self.shape)
        self.assertEqual(result['visible_step_count'], 1)
        self.assertNotEqual(result['edge_stages']['deduplicated'], result['edge_stages']['refined'])

    def test_empty_inputs(self):
        self.assertEqual(count_visible_steps([], [], self.shape)['visible_step_count'], 0)

    def test_cropped_edge_does_not_participate_in_gap_spacing(self):
        cropped = {'class': 0, 'confidence': 0.9, 'endpoints': [[20, 0], [139, 0]]}
        lines = [cropped, *(self.line(y) for y in (25, 45, 85))]
        result = count_visible_steps([], lines, self.shape)
        self.assertEqual(result['visible_step_count'], 4)
        self.assertEqual(len(result['suspected_gaps']), 1)

    def test_groups_shifted_edges_when_local_spacing_and_surface_agree(self):
        lines = [self.line(y) for y in (20, 40, 44, 65)]
        result = count_grouped_steps([(1, 0.9, self.mask(42, 55))], lines, self.shape)
        self.assertEqual(len(result['edge_stages']['deduplicated']), 4)
        self.assertEqual(result['visible_step_count'], 3)
        self.assertEqual(len(result['steps'][1]['edge_segments']), 2)

    def test_preserves_nearby_edges_that_match_different_risers(self):
        surfaces = [(1, 0.9, self.mask(y, y + 3)) for y in (40, 44)]
        result = count_grouped_steps(surfaces, [self.line(40), self.line(44)], self.shape)
        self.assertEqual(result['visible_step_count'], 2)

    def test_corner_needs_a_shared_surface(self):
        lines = [{'class': 0, 'confidence': 0.9, 'endpoints': [[20, 50], [75, 50]]},
                 {'class': 0, 'confidence': 0.9, 'endpoints': [[76, 50], [134, 79]]}]
        riser = np.zeros(self.shape, dtype=bool)
        for x in range(20, 135):
            y = 50 + max(0, x - 76) // 2
            riser[y:y + 10, x] = True
        self.assertEqual(count_grouped_steps([], lines, self.shape)['visible_step_count'], 2)
        self.assertEqual(count_grouped_steps([(1, 0.9, riser)], lines, self.shape)['visible_step_count'], 1)

    def test_repeating_spacing_protects_close_distinct_levels(self):
        lines = [self.line(y) for y in (30, 34, 38, 42)]
        surfaces = [(1, 0.9, self.mask(36, 45))]
        old = count_grouped_steps(surfaces, lines, self.shape, version='2.2')
        new = count_grouped_steps(surfaces, lines, self.shape)
        self.assertEqual(old['visible_step_count'], 3)
        self.assertEqual(new['visible_step_count'], 4)
        self.assertIn('repeated_adjacent_spacing', [event['reason'] for event in new['blocked_merges']])

    def test_corner_matching_uses_order_of_multiple_levels(self):
        lines = []
        for y in (35, 65):
            lines.extend([{'class': 0, 'confidence': 0.9, 'endpoints': [[20, y], [75, y]]},
                          {'class': 0, 'confidence': 0.9, 'endpoints': [[76, y], [134, y + 29]]}])
        self.assertEqual(count_grouped_steps([], lines, self.shape, version='2.2')['visible_step_count'], 4)
        result = count_grouped_steps([], lines, self.shape)
        self.assertEqual(result['visible_step_count'], 2)
        self.assertEqual([event['reason'] for event in result['merge_events']], ['corner_repeated_order'] * 2)

    def test_shared_endpoint_can_group_skewed_predictions_with_spacing_evidence(self):
        endpoints = [[[96, 352], [501, 382]], [[109, 344], [422, 367]],
                     [[175, 242], [500, 242]], [[225, 150], [500, 150]]]
        lines = [{'class': 0, 'confidence': 0.9, 'endpoints': points} for points in endpoints]
        old = count_grouped_steps([], lines, (512, 512), version='2.3')
        result = count_grouped_steps([], lines, (512, 512))
        self.assertEqual(old['visible_step_count'], 4)
        self.assertEqual(result['visible_step_count'], 3)
        self.assertEqual(result['merge_events'][-1]['reason'], 'shared_endpoint_local_spacing')

    def test_shared_endpoint_without_other_levels_does_not_merge(self):
        lines = [{'class': 0, 'confidence': 0.9, 'endpoints': points}
                 for points in ([[96, 352], [501, 382]], [[109, 344], [422, 367]])]
        result = count_grouped_steps([], lines, (512, 512))
        self.assertEqual(result['visible_step_count'], 2)
