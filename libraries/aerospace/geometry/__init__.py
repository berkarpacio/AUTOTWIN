from .airframe_fw import create_aircraft_geometry

__all__ = ["create_aircraft_geometry", "configurator"]


def __getattr__(name):
    # The configurator module is not present in every checkout. Import it only
    # when requested, so independent geometry definitions remain usable.
    if name == "configurator":
        from .configurator import configurator
        return configurator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
