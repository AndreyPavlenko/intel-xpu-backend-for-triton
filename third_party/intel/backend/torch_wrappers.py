import torch
from weakref import WeakValueDictionary

# This dictionary is used for the case, when a tensor is passed as a pointer to the kernel.
_ptr_to_tensor = WeakValueDictionary()


def wrap_launch(mod, *args):
    """
    Copy tensors to the device before launching and back after launching.
    """
    tensors = {}
    new_args = list(args[:9])

    def process_arg(arg):
        if isinstance(arg, tuple):
            return tuple(process_arg(a) for a in arg)

        is_ptr = False
        if isinstance(arg, int):
            if (tensor := _ptr_to_tensor.get(arg, None)) is not None:
                arg = tensor
                is_ptr = True
            else:
                return arg

        if isinstance(tensor := getattr(arg, "base", None), torch.Tensor):  # triton.runtime.jit.TensorWrapper
            arg = tensor
        if isinstance(arg, torch.Tensor) and ((dev := getattr(arg, "_xpu_dev", None)) or not arg.is_xpu):
            if (dev_tensor := tensors.get(arg, None)) is None:
                dev_tensor = arg.to(dev or "xpu")
            tensors[arg] = dev_tensor
            return dev_tensor.data_ptr() if is_ptr else dev_tensor
        return arg

    for arg in args[9:]:
        new_args.append(process_arg(arg))

    mod.launch(tuple(new_args))

    for cpu_tensor, dev_tensor in tensors.items():
        cpu_tensor.copy_(dev_tensor)


class _DeviceWrapper:
    """
    A device wrapper, that overrides the comparison operators.
    """

    def __init__(self, device):
        assert device.type == "xpu" or device.type == "cpu"
        self._device = device

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __eq__(self, other):
        return isinstance(other, (_DeviceWrapper, torch.device)) and (other.type == "xpu" or other.type == "cpu")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self._device)

    def __str__(self):
        return str(self._device)


def _attach_device(tensor, device):
    """
    Save the device in the tensor's `_xpu_dev` attribute.
    """
    if isinstance(tensor, torch.Tensor) and not hasattr(tensor, "_xpu_dev"):
        if isinstance(device, str):
            device = torch.device(device, 0)
        elif isinstance(device, _DeviceWrapper):
            device = device._device
        tensor._xpu_dev = device
        _ptr_to_tensor[tensor.data_ptr()] = tensor


def _device_arg_decorator(func):
    """
    If the function has a `device` argument and the type is "xpu", then remove the argument, call the function and
    attach the device to the resulting tensor.
    """

    def wrapper(*args, **kwargs):
        if device := kwargs.get("device", None):
            if device == "xpu" or getattr(device, "type", None) == "xpu":
                kwargs.pop("device")
                tensor = func(*args, **kwargs)
                _attach_device(tensor, device)
                return tensor
        return func(*args, **kwargs)

    return wrapper


def _xpu_dev_decorator(func, idx):
    """
    If the argument at the specified index has the `_xpu_dev` attribute, then attach it to the resulting tensor.
    """

    def wrapper(*args, **kwargs):
        tensor = func(*args, **kwargs)
        if len(args) > idx and (dev := getattr(args[idx], "_xpu_dev", None)):
            _attach_device(tensor, dev)
        return tensor

    return wrapper


def _xpu_dev_property_decorator(prop):
    """
    A property decorator, that returns a `_DeviceWrapper` for xpu and cpu devices.
    """

    @property
    def wrapper(self):
        if dev := getattr(self, "_xpu_dev", None):
            return _DeviceWrapper(dev)
        dev = prop.__get__(self)
        return _DeviceWrapper(dev) if dev.type in ("xpu", "cpu") else dev

    return wrapper


def _ignore_err_decorator(func):
    """
    Silently ignore errors in the decorated function.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass

    return wrapper


# The following functions are decorated to return a cpu tensor with the `_xpu_dev` attribute,
# if the "xpu" device is specified in the function arguments.
for name in (
        "arange",
        "as_tensor",
        "asarray",
        "bartlett_window",
        "blackman_window",
        "empty",
        "empty_like",
        "empty_permuted",
        "empty_strided",
        "eye",
        "full",
        "full_like",
        "hamming_window",
        "hann_window",
        "kaiser_window",
        "linspace",
        "logspace",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "range",
        "sparse_bsc_tensor",
        "sparse_bsr_tensor",
        "sparse_compressed_tensor",
        "sparse_coo_tensor",
        "sparse_csc_tensor",
        "sparse_csr_tensor",
        "tensor",
        "tril_indices",
        "triu_indices",
        "zeros",
        "zeros_like",
):
    setattr(torch, name, _device_arg_decorator(getattr(torch, name)))

# The following functions are decorated to propagate the `_xpu_dev` attribute from the input tensor to the output.
for name in (
        "as_tensor",
        "asarray",
        "empty_like",
        "full_like",
        "ones_like",
        "rand_like",
        "randint_like",
        "randn_like",
        "zeros_like",
):
    setattr(torch, name, _xpu_dev_decorator(getattr(torch, name), 0))

# Override Tensor.device to return a `_DeviceWrapper`.
torch.Tensor.device = _xpu_dev_property_decorator(torch.Tensor.device)

# The Event.record() function fails with UR_RESULT_ERROR_UNSUPPORTED_FEATURE
torch.xpu.Event.record = _ignore_err_decorator(torch.xpu.Event.record)
