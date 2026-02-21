import warnings

warnings.filterwarnings(
	"ignore",
	message=r"`torch\.jit\.script` is deprecated\..*",
	category=DeprecationWarning,
	module=r"torch\.jit\._script",
	append=True,
)
