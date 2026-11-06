import asyncio
import threading
import unittest
import weakref

from websockets.exceptions import CloseRejected, SendRejected
from websockets.policy import (
    CloseGate,
    PolicyFrame,
    SendAdmit,
    WaitGate,
    admit_send,
    current_close_gate,
    current_policy,
    effective_wait_timeout,
    policy_scope,
    pop_policy,
    push_policy,
    send_can_continue,
    stacks_context,
)


class FakeConnection:
    pass


class PolicyStackTests(unittest.TestCase):
    def setUp(self):
        self.connection = FakeConnection()
        self.other_connection = FakeConnection()

    def tearDown(self):
        stacks_context.set(None)

    def test_frame_defaults(self):
        frame = push_policy(self.connection)
        self.assertEqual(frame.send_admit, SendAdmit.HONOR_CFG)
        self.assertEqual(frame.close_gate, CloseGate.HONOR_CFG)
        self.assertEqual(frame.wait_gate, WaitGate.HONOR_CFG)
        self.assertIsNone(frame.shrink)
        self.assertFalse(frame.one_used)
        pop_policy(self.connection, frame)

    def test_default_view_without_frame(self):
        self.assertIsNone(current_policy(self.connection))
        self.assertEqual(effective_wait_timeout(self.connection, 10.0), (10.0, False))
        self.assertEqual(effective_wait_timeout(self.connection, None), (None, False))
        self.assertEqual(current_close_gate(self.connection), CloseGate.HONOR_CFG)
        admit_send(self.connection)  # doesn't raise
        send_can_continue(self.connection)  # doesn't raise

    def test_push_shows_frame_at_top(self):
        frame = push_policy(self.connection, send_admit=SendAdmit.DENY_NEW)
        self.assertIs(current_policy(self.connection), frame)
        pop_policy(self.connection, frame)
        self.assertIsNone(current_policy(self.connection))

    def test_pop_without_push(self):
        frame = PolicyFrame()
        with self.assertRaises(RuntimeError):
            pop_policy(self.connection, frame)

    def test_pop_out_of_order(self):
        outer = push_policy(self.connection, send_admit=SendAdmit.DENY_NEW)
        inner = push_policy(self.connection, send_admit=SendAdmit.ONE_MORE)
        with self.assertRaises(RuntimeError):
            pop_policy(self.connection, outer)
        pop_policy(self.connection, inner)
        pop_policy(self.connection, outer)

    def test_pop_unrelated_connection(self):
        frame = push_policy(self.connection)
        with self.assertRaises(RuntimeError):
            pop_policy(self.other_connection, frame)
        pop_policy(self.connection, frame)

    def test_nested_frames_restore_outer_view(self):
        outer = push_policy(self.connection, send_admit=SendAdmit.DENY_NEW)
        self.assertIs(current_policy(self.connection), outer)
        inner = push_policy(self.connection, close_gate=CloseGate.DENY_CLOSE)
        self.assertIs(current_policy(self.connection), inner)
        pop_policy(self.connection, inner)
        self.assertIs(current_policy(self.connection), outer)
        pop_policy(self.connection, outer)
        self.assertIsNone(current_policy(self.connection))

    def test_policy_scope_pushes_and_pops(self):
        with policy_scope(self.connection, send_admit=SendAdmit.DENY_NEW) as frame:
            self.assertIs(current_policy(self.connection), frame)
            with self.assertRaises(SendRejected):
                admit_send(self.connection)
        self.assertIsNone(current_policy(self.connection))
        admit_send(self.connection)  # restored behavior

    def test_policy_scope_pops_on_exception(self):
        with self.assertRaises(RuntimeError):
            with policy_scope(self.connection, close_gate=CloseGate.DENY_CLOSE):
                raise RuntimeError("boom")
        self.assertIsNone(current_policy(self.connection))

    def test_empty_stack_is_honored(self):
        # White-box check: an empty stack behaves like no stack at all.
        mapping = weakref.WeakKeyDictionary()
        mapping[self.connection] = []
        token = stacks_context.set(mapping)
        try:
            self.assertIsNone(current_policy(self.connection))
            self.assertEqual(
                effective_wait_timeout(self.connection, 10.0),
                (10.0, False),
            )
        finally:
            stacks_context.reset(token)

    def test_frames_are_per_connection(self):
        push_policy(self.connection, send_admit=SendAdmit.DENY_NEW)
        # The other connection doesn't have a frame pushed for it.
        self.assertIsNone(current_policy(self.other_connection))
        admit_send(self.other_connection)  # doesn't raise
        with self.assertRaises(SendRejected):
            admit_send(self.connection)

    def test_deny_new_rejects_every_send(self):
        with policy_scope(self.connection, send_admit=SendAdmit.DENY_NEW):
            for _ in range(3):
                with self.assertRaises(SendRejected):
                    admit_send(self.connection)

    def test_deny_new_continuation_gate(self):
        with policy_scope(self.connection, send_admit=SendAdmit.DENY_NEW):
            with self.assertRaises(SendRejected):
                send_can_continue(self.connection)

    def test_one_more_admits_exactly_one_send(self):
        with policy_scope(self.connection, send_admit=SendAdmit.ONE_MORE):
            admit_send(self.connection)  # the one allowed send
            with self.assertRaises(SendRejected):
                admit_send(self.connection)
            with self.assertRaises(SendRejected):
                admit_send(self.connection)

    def test_one_more_fragments_dont_consume_more(self):
        with policy_scope(self.connection, send_admit=SendAdmit.ONE_MORE) as frame:
            admit_send(self.connection)
            self.assertTrue(frame.one_used)
            # Continuation fragments of the admitted send aren't rejected.
            send_can_continue(self.connection)

    def test_honor_cfg_admit(self):
        with policy_scope(self.connection, send_admit=SendAdmit.HONOR_CFG):
            admit_send(self.connection)
            admit_send(self.connection)
            send_can_continue(self.connection)

    def test_close_gate_views(self):
        with policy_scope(self.connection, close_gate=CloseGate.DENY_CLOSE):
            self.assertEqual(
                current_close_gate(self.connection),
                CloseGate.DENY_CLOSE,
            )
        with policy_scope(self.connection, close_gate=CloseGate.FORCE_CLOSE):
            self.assertEqual(
                current_close_gate(self.connection),
                CloseGate.FORCE_CLOSE,
            )

    def test_close_rejected_exception(self):
        with policy_scope(self.connection, close_gate=CloseGate.DENY_CLOSE):
            with self.assertRaises(CloseRejected):
                raise CloseRejected("denied")

    def test_wait_timeout_honor_cfg(self):
        with policy_scope(self.connection):
            self.assertEqual(
                effective_wait_timeout(self.connection, 10.0),
                (10.0, False),
            )

    def test_wait_timeout_freeze(self):
        with policy_scope(self.connection, wait_gate=WaitGate.FREEZE):
            self.assertEqual(
                effective_wait_timeout(self.connection, 10.0),
                (None, True),
            )
            self.assertEqual(
                effective_wait_timeout(self.connection, None),
                (None, True),
            )

    def test_wait_timeout_shrink(self):
        with policy_scope(
            self.connection,
            wait_gate=WaitGate.SHRINK_NOW,
            shrink=5.0,
        ):
            self.assertEqual(
                effective_wait_timeout(self.connection, 10.0),
                (5.0, True),
            )
            self.assertEqual(
                effective_wait_timeout(self.connection, None),
                (5.0, True),
            )

    def test_wait_timeout_nested_shrinks_keep_tightest(self):
        with policy_scope(
            self.connection,
            wait_gate=WaitGate.SHRINK_NOW,
            shrink=2.0,
        ):
            with policy_scope(
                self.connection,
                wait_gate=WaitGate.SHRINK_NOW,
                shrink=7.0,
            ):
                self.assertEqual(
                    effective_wait_timeout(self.connection, 10.0),
                    (2.0, True),
                )

    def test_wait_timeout_freeze_then_inner_shrink_reenables(self):
        with policy_scope(self.connection, wait_gate=WaitGate.FREEZE):
            with policy_scope(
                self.connection,
                wait_gate=WaitGate.SHRINK_NOW,
                shrink=3.0,
            ):
                self.assertEqual(
                    effective_wait_timeout(self.connection, 10.0),
                    (3.0, True),
                )
            # Popping the inner frame restores the outer freeze.
            self.assertEqual(
                effective_wait_timeout(self.connection, 10.0),
                (None, True),
            )

    def test_wait_timeout_shrink_then_inner_freeze(self):
        with policy_scope(
            self.connection,
            wait_gate=WaitGate.SHRINK_NOW,
            shrink=3.0,
        ):
            with policy_scope(self.connection, wait_gate=WaitGate.FREEZE):
                self.assertEqual(
                    effective_wait_timeout(self.connection, 10.0),
                    (None, True),
                )

    def test_shrink_requires_value(self):
        with self.assertRaises(ValueError):
            push_policy(self.connection, wait_gate=WaitGate.SHRINK_NOW)

    def test_shrink_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            push_policy(
                self.connection,
                wait_gate=WaitGate.SHRINK_NOW,
                shrink=-1.0,
            )

    def test_shrink_without_shrink_gate_rejected(self):
        with self.assertRaises(ValueError):
            push_policy(self.connection, shrink=1.0)

    def test_connection_doesnt_leak_frame_mapping(self):
        connection = FakeConnection()
        push_policy(connection, send_admit=SendAdmit.DENY_NEW)
        mapping = stacks_context.get()
        self.assertIsNotNone(mapping)
        self.assertIn(connection, mapping)
        del connection
        # The weak key disappears when the connection is garbage collected.
        self.assertEqual(list(mapping), [])


class ContextIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        stacks_context.set(None)

    async def test_task_created_in_scope_inherits_snapshot(self):
        connection = FakeConnection()
        with policy_scope(connection, send_admit=SendAdmit.DENY_NEW):

            async def child():
                with self.assertRaises(SendRejected):
                    admit_send(connection)

            await asyncio.create_task(child())

    async def test_concurrent_tasks_dont_share_stacks(self):
        connection = FakeConnection()
        entered = asyncio.Event()

        async def scope_deny():
            with policy_scope(connection, send_admit=SendAdmit.DENY_NEW):
                entered.set()
                await asyncio.sleep(0)
                with self.assertRaises(SendRejected):
                    admit_send(connection)

        async def scope_honor():
            await entered.wait()
            # This task has no frame for the connection.
            self.assertIsNone(current_policy(connection))
            admit_send(connection)

        await asyncio.gather(scope_deny(), scope_honor())
        # The parent task isn't affected by either child task.
        self.assertIsNone(current_policy(connection))

    async def test_child_pop_doesnt_change_parent(self):
        connection = FakeConnection()
        outer = push_policy(connection, send_admit=SendAdmit.DENY_NEW)

        async def child():
            # The child owns a copy of the context after pushing.
            with policy_scope(connection, send_admit=SendAdmit.ONE_MORE):
                self.assertIs(
                    current_policy(connection).send_admit,
                    SendAdmit.ONE_MORE,
                )

        await asyncio.create_task(child())
        # The parent still sees its own frame.
        self.assertIs(current_policy(connection), outer)
        pop_policy(connection, outer)

    async def test_per_connection_isolation_in_same_task(self):
        first = FakeConnection()
        second = FakeConnection()
        with policy_scope(first, send_admit=SendAdmit.DENY_NEW):
            with self.assertRaises(SendRejected):
                admit_send(first)
            self.assertIsNone(current_policy(second))
            admit_send(second)


class ThreadIsolationTests(unittest.TestCase):
    def tearDown(self):
        stacks_context.set(None)

    def test_threads_dont_share_stacks(self):
        connection = FakeConnection()
        results = {}

        with policy_scope(connection, send_admit=SendAdmit.DENY_NEW):

            def child():
                # A new thread starts with a fresh execution context.
                results["sees_frame"] = current_policy(connection) is not None
                admit_send(connection)
                results["admitted"] = True
                with policy_scope(connection, close_gate=CloseGate.DENY_CLOSE):
                    results["gate"] = current_close_gate(connection)

            thread = threading.Thread(target=child)
            thread.start()
            thread.join()

        self.assertFalse(results["sees_frame"])
        self.assertTrue(results["admitted"])
        self.assertEqual(results["gate"], CloseGate.DENY_CLOSE)
        # The child frame never escaped into the parent thread.
        self.assertIsNone(current_policy(connection))


if __name__ == "__main__":
    unittest.main()
