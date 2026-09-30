from mcreid.topology import CameraTopology, TransitionWindow


def test_unknown_transition_allowed_by_default():
    assert CameraTopology().allows("a", "b", 3.0)


def test_strict_topology_rejects_unknown_transition():
    assert not CameraTopology(strict=True).allows("a", "b", 3.0)


def test_transition_window():
    topology = CameraTopology(
        {"a": {"b": TransitionWindow(min_seconds=5, max_seconds=10)}}
    )
    assert not topology.allows("a", "b", 4.9)
    assert topology.allows("a", "b", 7.0)
    assert not topology.allows("a", "b", 10.1)
