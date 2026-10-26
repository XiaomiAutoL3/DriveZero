"""nuPlan v10 Teacher entrypoint."""

from importlib import import_module


def get_teacher_class(name: str):
    if name != "nuplan_v10":
        raise ValueError("This release supports only teacher.name=nuplan_v10.")
    return import_module("teachers.nuplan_v10").Teacher


__all__ = ["get_teacher_class"]
