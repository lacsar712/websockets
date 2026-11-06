"""
Runtime policy frames for WebSocket connections.

This module provides a stack of policy frames bound to the current execution
context — a task in the :mod:`asyncio` implementation or a thread in the
:mod:`threading` implementation. Frames are also bound to a specific
connection, so a frame pushed for one connection never affects another
connection used in the same context.

Policy frames make it possible to change, for the duration of a scope, how
sends, active closes, and waits behave, without changing the permanent
configuration of the connection (``open_timeout``, ``close_timeout``,
``max_size``, ``max_queue``, ...) and without storing state on the connection.

A frame controls three independent axes:

* ``send_admit`` decides whether new :meth:`send` calls are admitted:
  :attr:`SendAdmit.HONOR_CFG` (the default),
  :attr:`SendAdmit.DENY_NEW`, or :attr:`SendAdmit.ONE_MORE`.
* ``close_gate`` decides what an active :meth:`close` does:
  :attr:`CloseGate.HONOR_CFG` (the default),
  :attr:`CloseGate.DENY_CLOSE`, or :attr:`CloseGate.FORCE_CLOSE`.
* ``wait_gate`` decides how waits are bounded:
  :attr:`WaitGate.HONOR_CFG` (the default),
  :attr:`WaitGate.FREEZE`, or :attr:`WaitGate.SHRINK_NOW`.

Use :func:`policy_scope` as a context manager, or :func:`push_policy` and
:func:`pop_policy` directly. Nested callbacks and nested scopes always read
the top of the stack at the time of the call; popping a frame restores the
view provided by outer frames.

"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .exceptions import SendRejected


__all__ = [
    "PolicyFrame",
    "SendAdmit",
    "CloseGate",
    "WaitGate",
    "push_policy",
    "pop_policy",
    "policy_scope",
    "current_policy",
    "effective_wait_timeout",
    "admit_send",
    "send_can_continue",
    "current_close_gate",
]


class SendAdmit(enum.Enum):
    """Whether new sends are admitted by a policy frame."""

    HONOR_CFG = "honor_cfg"
    """Behave as without the policy frame."""

    DENY_NEW = "deny_new"
    """Reject every new send; sends already in progress are not affected."""

    ONE_MORE = "one_more"
    """Admit exactly one more send, then reject further sends."""


class CloseGate(enum.Enum):
    """What an active close does under a policy frame."""

    HONOR_CFG = "honor_cfg"
    """Perform the usual closing handshake."""

    DENY_CLOSE = "deny_close"
    """Reject active closes; closes initiated by the peer are not affected."""

    FORCE_CLOSE = "force_close"
    """Fail the connection and terminate the transport immediately."""


class WaitGate(enum.Enum):
    """How waits are bounded under a policy frame."""

    HONOR_CFG = "honor_cfg"
    """Use the timeout configured on the connection or by the call."""

    FREEZE = "freeze"
    """Do not fail the wait on a timeout while the frame is in effect."""

    SHRINK_NOW = "shrink_now"
    """Bound the wait by the ``shrink`` value of the frame."""


@dataclass
class PolicyFrame:
    """
    Policy frame pushed on the runtime policy stack of a connection.

    Frames are created by :func:`push_policy` and :func:`policy_scope`.

    """

    send_admit: SendAdmit = SendAdmit.HONOR_CFG
    close_gate: CloseGate = CloseGate.HONOR_CFG
    wait_gate: WaitGate = WaitGate.HONOR_CFG
    shrink: float | None = None
    one_used: bool = field(default=False, init=False)
    """Whether the single send allowed by :attr:`SendAdmit.ONE_MORE` ran."""


# Stacks of frames, keyed by connection, in the current execution context.
# The mapping is replaced on every push so that concurrent tasks and threads
# never observe frames pushed in another context; within a context it is then
# mutated in place so that nested calls read the current stack dynamically.
stacks_context: contextvars.ContextVar[
    weakref.WeakKeyDictionary[Any, list[PolicyFrame]] | None
] = contextvars.ContextVar("policy_stacks", default=None)


def push_policy(
    connection: Any,
    *,
    send_admit: SendAdmit = SendAdmit.HONOR_CFG,
    close_gate: CloseGate = CloseGate.HONOR_CFG,
    wait_gate: WaitGate = WaitGate.HONOR_CFG,
    shrink: float | None = None,
) -> PolicyFrame:
    """
    Push a policy frame for ``connection`` in the current execution context.

    Args:
        connection: Connection governed by the frame.
        send_admit: Admission policy for new :meth:`send` calls.
        close_gate: Policy for active :meth:`close` calls.
        wait_gate: Policy for close, opening, and receive waits.
        shrink: Upper bound in seconds for waits when
            ``wait_gate`` is :attr:`WaitGate.SHRINK_NOW`.

    Returns:
        The pushed frame, to pass to :func:`pop_policy`.

    Raises:
        ValueError: If ``shrink`` is missing for :attr:`WaitGate.SHRINK_NOW`
            or provided for another wait policy.

    """
    if wait_gate is WaitGate.SHRINK_NOW:
        if shrink is None:
            raise ValueError("shrink is required when wait_gate is SHRINK_NOW")
        if shrink < 0:
            raise ValueError("shrink must be positive or equal to zero")
    elif shrink is not None:
        raise ValueError("shrink applies only when wait_gate is SHRINK_NOW")

    frame = PolicyFrame(
        send_admit=send_admit,
        close_gate=close_gate,
        wait_gate=wait_gate,
        shrink=shrink,
    )

    stacks = stacks_context.get()
    if stacks is None:
        stacks = weakref.WeakKeyDictionary()
    else:
        # Copy-on-write: detach from the context (the parent task or thread)
        # that this mapping was inherited from.
        stacks = stacks.copy()
    stack = list(stacks.get(connection, ()))
    stack.append(frame)
    stacks[connection] = stack
    stacks_context.set(stacks)
    return frame


def pop_policy(connection: Any, frame: PolicyFrame) -> None:
    """
    Pop ``frame`` for ``connection`` from the stack in the current context.

    Args:
        connection: Connection governed by the frame.
        frame: Frame returned by :func:`push_policy`.

    Raises:
        RuntimeError: If no frame is pushed for ``connection`` or if ``frame``
            isn't at the top of the stack.

    """
    stacks = stacks_context.get()
    if stacks is None:
        raise RuntimeError("cannot pop policy frame: no policy frame is pushed")
    stack = stacks.get(connection)
    if not stack or stack[-1] is not frame:
        raise RuntimeError("cannot pop policy frame: it isn't at the top of the stack")
    stack.pop()
    if not stack:
        del stacks[connection]


@contextlib.contextmanager
def policy_scope(
    connection: Any,
    *,
    send_admit: SendAdmit = SendAdmit.HONOR_CFG,
    close_gate: CloseGate = CloseGate.HONOR_CFG,
    wait_gate: WaitGate = WaitGate.HONOR_CFG,
    shrink: float | None = None,
) -> Iterator[PolicyFrame]:
    """
    Push a policy frame for the duration of the ``with`` block.

    The frame is popped when exiting the block, including when an exception
    is raised. Arguments are identical to :func:`push_policy`.

    """
    frame = push_policy(
        connection,
        send_admit=send_admit,
        close_gate=close_gate,
        wait_gate=wait_gate,
        shrink=shrink,
    )
    try:
        yield frame
    finally:
        pop_policy(connection, frame)


def current_policy(connection: Any) -> PolicyFrame | None:
    """
    Return the policy frame currently at the top of ``connection``'s stack.

    Returns :obj:`None` when no frame is pushed, which makes every operation
    behave as without any policy.

    """
    stacks = stacks_context.get()
    if stacks is None:
        return None
    stack = stacks.get(connection)
    if not stack:
        return None
    return stack[-1]


def effective_wait_timeout(
    connection: Any,
    configured: float | None,
) -> tuple[float | None, bool]:
    """
    Merge a configured wait timeout with the wait policy of active frames.

    Frames are applied from the outermost to the innermost:
    :attr:`WaitGate.HONOR_CFG` leaves the timeout unchanged,
    :attr:`WaitGate.FREEZE` removes it, and :attr:`WaitGate.SHRINK_NOW`
    tightens it to the frame's bound (and re-enables a timeout inside an
    outer :attr:`~WaitGate.FREEZE` frame).

    Args:
        connection: Connection whose waits are being bounded.
        configured: Timeout in seconds configured on the connection or by
            the call, or :obj:`None` for no limit.

    Returns:
        ``(timeout, governed)``: the effective timeout and whether any frame
        changed the view. When ``governed`` is false, callers keep their
        existing behavior exactly.

    """
    stacks = stacks_context.get()
    if stacks is None:
        return configured, False
    stack = stacks.get(connection)
    if not stack:
        return configured, False
    timeout = configured
    governed = False
    for frame in stack:
        if frame.wait_gate is WaitGate.FREEZE:
            timeout = None
            governed = True
        elif frame.wait_gate is WaitGate.SHRINK_NOW:
            assert frame.shrink is not None
            timeout = frame.shrink if timeout is None else min(timeout, frame.shrink)
            governed = True
    return timeout, governed


def admit_send(connection: Any) -> None:
    """
    Check the send admission policy before starting a new send.

    Raises:
        SendRejected: If the frame at the top of the stack rejects new sends
            or if the single send allowed by :attr:`SendAdmit.ONE_MORE`
            already ran.

    """
    frame = current_policy(connection)
    if frame is None:
        return
    if frame.send_admit is SendAdmit.DENY_NEW:
        raise SendRejected("sending a new message is denied by the policy in effect")
    if frame.send_admit is SendAdmit.ONE_MORE:
        if frame.one_used:
            raise SendRejected("the one allowed send was already used")
        frame.one_used = True


def send_can_continue(connection: Any) -> None:
    """
    Check the send admission policy before sending another fragment.

    Unlike :func:`admit_send`, this doesn't consume the allowance of an
    :attr:`SendAdmit.ONE_MORE` frame: it only fails when a newly pushed
    :attr:`SendAdmit.DENY_NEW` frame forbids completing a fragmented send, in
    which case the fragmented send fails and the connection is closed.

    Raises:
        SendRejected: If the frame at the top of the stack denies new sends.

    """
    frame = current_policy(connection)
    if frame is not None and frame.send_admit is SendAdmit.DENY_NEW:
        raise SendRejected("sending the remaining fragments is denied by policy")


def current_close_gate(connection: Any) -> CloseGate:
    """
    Return the close gate at the top of ``connection``'s stack.

    """
    frame = current_policy(connection)
    if frame is None:
        return CloseGate.HONOR_CFG
    return frame.close_gate
