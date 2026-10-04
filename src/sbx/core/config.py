"""Строгая сборка неизменяемых конфигурационных объектов из словарей.

Оболочка читает YAML/JSON в `dict`, ядро превращает словарь в frozen dataclass и
валидирует структуру: неизвестные и недостающие поля дают `ConfigError` с путём до поля.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")


class ConfigError(ValueError):
    """Ошибка структуры конфигурации."""


def from_mapping(cls: type[T], data: Any, path: str = "$") -> T:
    """Собирает dataclass `cls` из словаря `data`, рекурсивно и строго."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: ожидался словарь, получено {type(data).__name__}")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{path}: неизвестные поля {unknown}; допустимые: {sorted(fields)}")

    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
        if name not in data:
            if has_default:
                continue
            raise ConfigError(f"{path}: отсутствует обязательное поле '{name}'")
        kwargs[name] = _convert(hints[name], data[name], f"{path}.{name}")
    return cls(**kwargs)


def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(arg, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: значение не подходит ни к одному типу: {errors}")
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"{path}: ожидалось одно из {list(args)}, получено {value!r}")
        return value
    if dataclasses.is_dataclass(tp):
        return from_mapping(tp, value, path)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{path}: ожидался список")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{path}: ожидалось {len(args)} элементов")
        return tuple(
            _convert(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if origin in (dict, Mapping) or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: ожидался словарь")
        key_tp, val_tp = args if args else (Any, Any)
        return types.MappingProxyType(
            {
                _convert(key_tp, k, f"{path}.<key>"): _convert(val_tp, v, f"{path}.{k}")
                for k, v in value.items()
            }
        )
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: ожидалось число, получено {value!r}")
        return float(value)
    if tp in (int, str, bool):
        if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
            raise ConfigError(f"{path}: ожидался {tp.__name__}, получено {value!r}")
        return value
    return value
