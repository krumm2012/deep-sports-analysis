"""Weak labels retain their provenance, weights, and recording-level boundaries."""
import numpy as np
import pytest

from dsa.measure.semantic_phases import PHASES
from dsa.measure.semantic_weak import balanced_source_weights, decode_ordered_phases, fit_model, group_evaluate


def test_source_weight_ratio_survives_class_balancing():
    y = np.array(['ready', 'ready', 'swing', 'swing', 'swing'])
    weights = np.array([1., .25, .25, .25, .25])
    balanced = balanced_source_weights(y, weights)
    assert np.isclose(balanced[y == 'ready'].sum(), balanced[y == 'swing'].sum())
    assert np.isclose(balanced[0]/balanced[1], 4.)


def test_scaler_and_classifier_fit_with_source_weights():
    x = np.array([[0., 2.], [1., 1.], [7., 3.], [8., 4.]])
    y = np.array(['ready', 'ready', 'swing', 'swing'])
    weights = np.array([1., .25, 1., .25])
    model = fit_model(x, y, weights)
    np.testing.assert_allclose(model.steps[0][1].mean_, np.average(x, weights=weights, axis=0))
    assert model.predict_proba(x).shape == (4, 2)


def test_group_checks_remove_both_human_and_weak_samples_of_heldout_clip():
    rng = np.random.default_rng(7)
    samples, x = [], []
    for group in range(4):
        for phase_index, phase in enumerate(PHASES):
            human = group == 0
            samples.append({'id': f'{group}-{phase_index}', 'clip_id': str(group), 'phase': phase,
                            'label_source': 'human' if human else 'weak_auto',
                            'human_reviewed': human, 'source_weight': 1. if human else .25})
            x.append(np.eye(5)[phase_index]+rng.normal(0, .05, 5))
    evaluation, model = group_evaluate(np.array(x), samples)
    assert set(model.classes_) == set(PHASES)
    assert evaluation['human_reference_check']['intervals'] == 5
    assert evaluation['weak_label_agreement']['intervals'] == 15
    for fold in evaluation['folds']:
        assert not set(fold['train_clips']) & set(fold['test_clips'])
        assert len(fold['test_ids']) == 5


def test_five_phase_decoder_keeps_order_and_minimum_duration():
    targets = np.repeat(np.arange(5), 8)
    probability = np.full((40, 5), .025)
    probability[np.arange(40), targets] = .9
    np.testing.assert_array_equal(decode_ordered_phases(probability), targets)
    with pytest.raises(ValueError, match='five ordered phases'):
        decode_ordered_phases(probability[:3])
