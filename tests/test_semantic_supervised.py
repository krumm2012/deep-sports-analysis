import numpy as np
import pytest

from dsa.measure.semantic_supervised import SupervisedPhaseModel, predict_frames, temporal_folds


def test_temporal_blocks_never_overlap_and_enforce_time_guard():
    clips = {str(i): {'split': 'train', 'start_seconds': i*10, 'frames': 250, 'fps': 25} for i in range(24)}
    folds = temporal_folds(clips, folds=4, guard_seconds=25)
    assert sorted(i for f in folds for i in f['test_clips']) == sorted(clips)
    for fold in folds:
        assert not set(fold['train_clips']) & set(fold['test_clips'])
        assert not set(fold['train_clips']) & set(fold['guarded_out_clips'])
        for a in fold['train_clips']:
            for b in fold['test_clips']:
                assert abs(clips[a]['start_seconds']-clips[b]['start_seconds'])-10 >= 25
    clips['v'] = {'split': 'validation', 'start_seconds': 999, 'frames': 250, 'fps': 25}
    with pytest.raises(ValueError, match='Only training'):
        temporal_folds(clips)


class ConstantScores:
    def __init__(self, classes, probability):
        self.classes_ = np.array(classes)
        self.probability = np.array(probability)

    def scores(self, x):
        p = np.tile(self.probability, (len(x), 1))
        return self.classes_[p.argmax(1)], p.max(1), p, np.ones(len(x), dtype=bool), np.full(len(x), .01)


def test_forehand_expert_only_redistributes_core_mass_for_known_types():
    model = SupervisedPhaseModel()
    model.shared = ConstantScores(range(5), [.1, .3, .25, .2, .15])
    model.classes_ = np.arange(5)
    model.experts = {'正手': ConstantScores([1, 2, 3], [.1, .8, .1])}
    pred, _, p, _, _ = model.scores(np.zeros((2, 192)), np.array(['正手', '无法判断']))
    np.testing.assert_allclose(p.sum(1), 1.)
    np.testing.assert_allclose(p[:, [0, 4]], [[.1, .15], [.1, .15]])
    np.testing.assert_allclose(p[0, 1:4], [.075, .6, .075])
    np.testing.assert_allclose(p[1], model.shared.probability)
    assert pred.tolist() == [2, 1]


def test_steady_human_preparation_is_not_rejected_only_for_low_motion():
    shared = ConstantScores(range(5), [.9, .025, .025, .025, .025])
    class Model:
        def scores(self, x, kind): return shared.scores(x)
    features = (np.zeros((30, 192)), np.ones(30, dtype=bool), np.ones(30, dtype=bool))
    pred, _, _, accepted, _ = predict_frames(Model(), features, '无法判断', 1.)
    assert (pred == 0).all() and accepted.all()
