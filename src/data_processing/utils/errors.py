class ADPError(Exception):
    """Base error for the Alzheimer data processing package."""


class ConfigError(ADPError):
    pass


class ValidationError(ADPError):
    pass


class DependencyError(ADPError):
    pass
