from mcreid.embedding import AppearanceEmbedder


class _CudaAvailable:
    @staticmethod
    def is_available():
        return True


class _CudaUnavailable:
    @staticmethod
    def is_available():
        return False


class _TorchGPU:
    cuda = _CudaAvailable()


class _TorchCPU:
    cuda = _CudaUnavailable()


def test_numeric_device_is_mapped_to_pytorch_cuda_device():
    assert AppearanceEmbedder._resolve_device("0", _TorchGPU()) == "cuda:0"
    assert AppearanceEmbedder._resolve_device("2", _TorchGPU()) == "cuda:2"


def test_auto_device_uses_cuda_when_available_and_cpu_otherwise():
    assert AppearanceEmbedder._resolve_device("auto", _TorchGPU()) == "cuda:0"
    assert AppearanceEmbedder._resolve_device("auto", _TorchCPU()) == "cpu"


def test_explicit_pytorch_device_is_preserved():
    assert AppearanceEmbedder._resolve_device("cuda:1", _TorchGPU()) == "cuda:1"
    assert AppearanceEmbedder._resolve_device("cpu", _TorchGPU()) == "cpu"
