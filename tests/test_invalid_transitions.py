import pytest
from mcreid.topology import CameraTopology
@pytest.mark.parametrize('gap',[-1.,float('nan'),float('inf'),-float('inf')])
def test_unknown_route_cannot_approve_impossible_time_gap(gap):
 assert not CameraTopology(strict=False).allows('a','b',gap)
 assert not CameraTopology(strict=False).allows('a','a',gap)
