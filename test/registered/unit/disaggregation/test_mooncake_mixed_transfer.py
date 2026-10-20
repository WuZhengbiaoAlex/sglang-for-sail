"""Mooncake mixed-transfer regressions with CPU pools and mocked transport."""

import struct
import threading
import unittest
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

import sglang.srt.disaggregation.mooncake.conn as mc
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import PrefillServerInfo
from sglang.srt.disaggregation.utils import set_draft_kv_transfer_metadata
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class EndMessages(BaseException):
    """Stop the synchronous bootstrap loop after consuming the fake socket."""


def run_loop(manager, method, messages, observe=None):
    # Execute the real thread body synchronously and record escaped exceptions.
    events = []

    def recv():
        if observe:
            observe()
        if not messages:
            raise EndMessages()
        return messages.pop(0)

    manager.server_socket = NS(recv_multipart=recv)

    def start(target):
        try:
            target()
        except EndMessages:
            pass
        except Exception as e:
            events.append(e)

    with patch.object(
        mc.threading,
        "Thread",
        side_effect=lambda target: NS(start=lambda: start(target)),
    ):
        method()
    return events


def manager(draft=2, tp=16):
    m = mc.MooncakeKVManager.__new__(mc.MooncakeKVManager)
    m.kv_args = KVArgs()
    a = m.kv_args
    for k, v in dict(
        page_size=64,
        engine_rank=0,
        kv_item_lens=[64, 1024, 1024],
        kv_data_ptrs=[10000, 20000, 30000],
        kv_layer_ids=[0, 1, 1],
        draft_kv_entry_count=draft,
        draft_kv_head_num=1,
        draft_total_kv_head_num=16,
        draft_kv_token_major=True,
    ).items():
        setattr(a, k, v)
    m.attn_tp_size = tp
    m.attn_cp_size = 1
    m.attn_cp_rank = 0
    m.attn_tp_rank = 0
    m.pp_size = 1
    m.pp_rank = 0
    m.dcp_size = 1
    m.dcp_rank = 0
    m.is_mla_backend = True
    m.is_hybrid_mla_backend = False
    m.enable_all_cp_ranks_for_transfer = False
    m.decode_kv_args_table = {}
    m.rejected_registrations = {}
    m.transfer_infos = {}
    m.req_to_decode_prefix_len = {}
    m.failed_sessions = set()
    m.session_failures = defaultdict(int)
    m.session_lock = threading.Lock()
    m.failure_lock = threading.Lock()
    m.failure_records = {}
    m.request_status = {}
    m.enable_staging = False
    m.sync_status_to_decode_endpoint = Mock()
    m._start_heartbeat_checker_thread = Mock()
    m.required_prefill_response_num_table = {}
    m.prefill_response_tracker = defaultdict(set)
    return m


def registration(session="ok", count=2, token_major=True):
    return [
        b"None",
        b"127.0.0.1",
        b"1234",
        session.encode(),
        struct.pack("3Q", 40000, 50000, 60000),
        b"",
        b"",
        b"0",
        b"8",
        b"64",
        b"",
        b"",
        struct.pack("3I", 0, 1, 1),
        b"",
        b"",
        b"",
        b"1",
        b"0",
        struct.pack("3Q", 64, 2048, 2048),
        str(count).encode(),
        b"2",
        b"16",
        b"64",
        b"1" if token_major else b"0",
    ]


def request(session, room):
    return [
        str(room).encode(),
        b"127.0.0.1",
        b"1234",
        session.encode(),
        np.array([2], dtype=np.int32).tobytes(),
        b"0",
        b"",
        b"1",
    ]


def pool(layout, heads=1):
    # Exercise the real allocator method and descriptors on CPU; no fake tensor layout.
    p = MHATokenToKVPool.__new__(MHATokenToKVPool)
    p.size = 128
    p.page_size = 64
    p.head_num = heads
    p.head_dim = 8
    p.v_head_dim = 8
    p.layer_num = 1
    p.store_dtype = torch.float16
    p.device = "cpu"
    p.kv_cache_layout = layout
    p._kv_vector_x = 8
    p.use_hnd = False
    p.enable_custom_mem_pool = False
    p.memory_saver_adapter = NS(region=lambda *a: nullcontext())
    p._create_buffers_normal()
    p._kv_buffer_descs = p._build_kv_buffer_descs()
    return p


class TestMooncakeMixedTransfer(CustomTestCase):
    def test_bootstrap_http_advertises_source_draft_presence(self):
        import asyncio
        import json
        from unittest.mock import AsyncMock

        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        for count in (0, 2):
            with patch.object(CommonKVBootstrapServer, "run"):
                server = CommonKVBootstrapServer("127.0.0.1", 0)
            data = dict(
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                pp_size=1,
                pp_rank=0,
                system_dp_size=1,
                system_dp_rank=0,
                rank_ip="127.0.0.1",
                rank_port=1234,
                page_size=64,
                kv_cache_dtype="float16",
                load_balance_method="follow_bootstrap_room",
                draft_kv_entry_count=count,
            )
            response = asyncio.run(
                server._handle_route_put(NS(json=AsyncMock(return_value=data)))
            )
            self.assertEqual(response.status, 200)
            response = asyncio.run(
                server._handle_route_get(
                    NS(
                        query={
                            k: "-1"
                            for k in [
                                "prefill_dp_rank",
                                "prefill_cp_rank",
                                "target_tp_rank",
                                "target_pp_rank",
                            ]
                        }
                    )
                )
            )
            info = PrefillServerInfo(**json.loads(response.text))
            self.assertEqual(getattr(info, "draft_kv_entry_count", None), count)

    def test_receiver_mapping_dummy_and_completion(self):
        for plain, hybrid, draft, source in [
            (True, False, 2, 2),
            (False, True, 2, 2),
            (True, False, 0, 0),
            (True, False, 2, 0),
            (False, True, 0, 0),
        ]:
            for rank in range(8):
                with self.subTest(
                    plain=plain, hybrid=hybrid, draft=draft, source=source, rank=rank
                ):
                    m = manager()
                    m.attn_tp_size = 8
                    m.kv_args.engine_rank = rank
                    m.kv_args.draft_kv_entry_count = draft
                    m.is_mla_backend = plain
                    m.is_hybrid_mla_backend = hybrid
                    info = PrefillServerInfo(16, 1, 1, 1, 64, None, True)
                    info.draft_kv_entry_count = source
                    m._resolve_rank_mapping(info)
                    expected = 1 if plain and not source else 2
                    self.assertEqual(info.required_prefill_response_num, expected)
                    r = mc.MooncakeKVReceiver.__new__(mc.MooncakeKVReceiver)
                    r.kv_mgr = m
                    r.prefill_info = info
                    for key in [
                        "target_tp_rank",
                        "target_tp_ranks",
                        "target_cp_ranks",
                        "target_pp_ranks",
                    ]:
                        setattr(r, key, getattr(info, key))
                    r.bootstrap_addr = "localhost"
                    r.prefill_dp_rank = 0
                    r.bootstrap_room = 99
                    r.conclude_state = None
                    m.connection_pool = {}
                    r._get_bootstrap_info_from_server = lambda dp, cp, tp, pp: dict(
                        rank_ip="localhost", rank_port=1000 + tp, tp=tp
                    )
                    r._register_kv_args = Mock(return_value=True)
                    r._setup_bootstrap_infos()
                    real = [b["tp"] for b in r.bootstrap_infos if not b["is_dummy"]]
                    self.assertEqual(
                        real, [2 * rank] if expected == 1 else [2 * rank, 2 * rank + 1]
                    )
                    m.request_status[99] = KVPoll.WaitingForInput
                    m.required_prefill_response_num_table[99] = expected
                    states = []
                    msgs = [
                        [b"99", str(int(KVPoll.Success)).encode(), str(tp).encode()]
                        for tp in real
                    ]
                    self.assertEqual(
                        run_loop(
                            m,
                            m.start_decode_thread,
                            msgs,
                            lambda: states.append(m.request_status[99]),
                        ),
                        [],
                    )
                    self.assertEqual(states[-1], KVPoll.Success)
                    if expected == 2:
                        self.assertNotEqual(states[1], KVPoll.Success)

    def test_invalid_registration_then_valid_and_request_failure(self):
        for malformed in [
            "counts",
            "bytes",
            "integer",
            "packed",
            "topology",
            "missing_layer",
            "one_sided_layers",
            "layer_count",
        ]:
            with self.subTest(malformed=malformed):
                m = manager()
                m.request_status = {1: KVPoll.Bootstrapping, 2: KVPoll.Bootstrapping}
                bad = registration("bad", count=1)
                if malformed == "bytes":
                    bad = registration("bad")
                    bad[4] = b"x"
                if malformed == "integer":
                    bad = registration("bad")
                    bad[20] = b"oops"
                if malformed == "packed":
                    bad = registration("bad", token_major=False)
                if malformed == "topology":
                    bad = registration("bad")
                    bad[17] = b"1"
                if malformed == "missing_layer":
                    bad = registration("bad")
                    bad[12] = struct.pack("3I", 9, 1, 1)
                if malformed == "one_sided_layers":
                    bad = registration("bad")
                    bad[12] = b""
                if malformed == "layer_count":
                    bad = registration("bad")
                    bad[12] = struct.pack("2I", 0, 1)
                messages = [bad, request("bad", 1), registration(), request("ok", 2)]
                errors = run_loop(m, m.start_prefill_thread, messages)
                self.assertEqual(errors, [])
                self.assertIn("ok", m.decode_kv_args_table)
                self.assertNotIn("bad", m.decode_kv_args_table)
                self.assertEqual(m.request_status[1], KVPoll.Failed)
                self.assertEqual(m.request_status[2], KVPoll.WaitingForInput)
                self.assertIn("Rejected Mooncake registration", m.failure_records[1])
                self.assertEqual(m.sync_status_to_decode_endpoint.call_count, 1)

    def test_staging_prefetch_uses_one_registration_lookup(self):
        class RemovedAfterRead(dict):
            def get(self, key, default=None):
                return self.pop(key, default)

        for present in (True, False):
            with self.subTest(present=present):
                m = manager()
                m.enable_staging = True
                m.kv_buffer_tensors = []
                m._staging_ctx = NS(prefetch_requested=set(), prefetch_sockets={})
                m.transfer_infos[1] = {"s": mc.TransferInfo.from_zmq(request("s", 1))}
                m.decode_kv_args_table = RemovedAfterRead(
                    {"s": NS(dst_attn_tp_size=8)} if present else {}
                )
                with patch.object(
                    mc, "get_schedule", return_value=NS(chunked_prefill_size=64)
                ), patch(
                    "sglang.srt.disaggregation.common.staging_handler.prefetch_staging_reqs"
                ) as prefetch:
                    m._prefetch_staging_reqs(1)
                self.assertEqual(prefetch.call_count, int(present))

    def test_registration_changes_do_not_terminate_transfer_worker(self):
        for change in ("reject", "recover", "missing"):
            with self.subTest(change=change):
                m = manager()
                m.enable_trace = False
                m.bootstrap_port = 8998
                m._staging_outstanding = defaultdict(int)
                m.send_kvcache_mixed = Mock(return_value=0)
                m.send_aux = Mock(return_value=0)
                for session, room in (("changing", 1), ("ok", 2)):
                    m._register_decode_peer(registration(session), session)
                    m.request_status[room] = KVPoll.WaitingForInput
                    m.transfer_infos[room] = {
                        session: mc.TransferInfo.from_zmq(request(session, room))
                    }
                if change == "recover":
                    m.rejected_registrations["changing"] = "layout rejected"
                elif change == "missing":
                    m.decode_kv_args_table.pop("changing")

                reached = threading.Event()
                resume = threading.Event()

                class InterleavedDict(dict):
                    """Pause a reader at a controlled registration-update boundary."""

                    def pause(self, key):
                        if key == "changing" and not reached.is_set():
                            reached.set()
                            if not resume.wait(10):
                                raise TimeoutError("Registration update did not finish")

                    def __contains__(self, key):
                        present = super().__contains__(key)
                        if change == "recover":
                            self.pause(key)
                        return present

                    def __getitem__(self, key):
                        if change == "reject":
                            self.pause(key)
                        return super().__getitem__(key)

                    def get(self, key, default=None):
                        if change == "reject":
                            self.pause(key)
                        value = super().get(key, default)
                        if change == "recover":
                            self.pause(key)
                        return value

                if change == "reject":
                    m.decode_kv_args_table = InterleavedDict(m.decode_kv_args_table)
                elif change == "recover":
                    m.rejected_registrations = InterleavedDict(m.rejected_registrations)

                chunks = [
                    NS(
                        room=room,
                        staging_counted=False,
                        is_last_chunk=True,
                        index_slice=slice(0, 1),
                        prefill_kv_indices=np.array([0]),
                        num_kv_tokens=1,
                        state_indices=[],
                        prefill_aux_index=0,
                    )
                    for room in (1, 2)
                ]
                errors = []

                def get_chunk():
                    if not chunks:
                        raise EndMessages()
                    return chunks.pop(0)

                def worker():
                    try:
                        m.transfer_worker(NS(get=get_chunk), None)
                    except EndMessages:
                        pass
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                try:
                    if change != "missing":
                        self.assertTrue(reached.wait(10), errors)
                        wire = registration(
                            "changing", count=1 if change == "reject" else 2
                        )
                        m._register_decode_peer(wire, "changing")
                finally:
                    resume.set()
                    thread.join(10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(m.request_status[1], KVPoll.Failed)
                self.assertEqual(m.request_status[2], KVPoll.Success)
                self.assertIn(1, m.failure_records)
                self.assertEqual(m.send_kvcache_mixed.call_count, 1)
                self.assertEqual(m.send_kvcache_mixed.call_args.args[0], "ok")
                self.assertEqual(m.send_aux.call_count, 1)
                self.assertNotIn(1, m.transfer_infos)
                self.assertNotIn(2, m.transfer_infos)
                self.assertFalse(m._staging_outstanding)
                statuses = [
                    (c.args[2], c.args[3])
                    for c in m.sync_status_to_decode_endpoint.call_args_list
                ]
                self.assertIn((1, KVPoll.Failed), statuses)
                self.assertIn((2, KVPoll.Success), statuses)
                self.assertNotIn((1, KVPoll.Success), statuses)

    def test_rejected_session_recovers_only_after_valid_registration(self):
        m = manager()
        m.request_status = {1: KVPoll.Bootstrapping}
        m.transfer_infos[1] = {"bad": mc.TransferInfo.from_zmq(request("bad", 1))}
        m.engine = NS(send_probe=lambda session: 0)
        m._register_decode_peer(registration("bad"), "bad")
        self.assertIn("bad", m.decode_kv_args_table)
        bad = registration("bad")
        bad[12] = struct.pack("3I", 9, 1, 1)
        errors = run_loop(m, m.start_prefill_thread, [bad])
        self.assertEqual(errors, [])
        self.assertEqual(m.request_status[1], KVPoll.Failed)
        m._run_one_probe_pass()
        self.assertIn("bad", m.rejected_registrations)
        self.assertNotIn("bad", m.decode_kv_args_table)
        errors = run_loop(m, m.start_prefill_thread, [registration("bad")])
        self.assertEqual(errors, [])
        self.assertIn("bad", m.decode_kv_args_table)
        self.assertNotIn("bad", m.rejected_registrations)

    def test_failed_room_is_not_revived_by_late_success(self):
        m = manager()
        m.request_status[1] = KVPoll.Failed
        m.required_prefill_response_num_table[1] = 1
        self.assertEqual(run_loop(m, m.start_decode_thread, [[b"1", b"4", b"0"]]), [])
        self.assertEqual(m.request_status[1], KVPoll.Failed)
        self.assertEqual(
            run_loop(m, m.start_prefill_thread, [registration(), request("ok", 1)]), []
        )
        self.assertEqual(m.request_status[1], KVPoll.Failed)
        self.assertNotIn(1, m.transfer_infos)

    def test_rejection_before_sender_creation_fails_without_timeout(self):
        import sglang.srt.disaggregation.common.conn as common

        m = manager()
        m.is_dummy_cp_rank = False
        m.enable_trace = False
        errors = run_loop(
            m, m.start_prefill_thread, [registration("bad", count=1), request("bad", 7)]
        )
        self.assertEqual(errors, [])
        self.assertIn(7, m.failure_records)
        with patch.object(common, "get_parallel", return_value=NS(dp_size=1)):
            sender = mc.MooncakeKVSender(m, "localhost", 7, [0], 0)
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_packed_layout_is_rejected_on_either_peer(self):
        for tp in [8, 16]:
            for side in ["source", "destination"]:
                with self.subTest(tp=tp, side=side):
                    m = manager(tp=tp)
                    if tp == 8:
                        m.kv_args.draft_kv_head_num = 2
                        m.kv_args.kv_item_lens = [64, 2048, 2048]
                    info = mc.KVArgsRegisterInfo.from_zmq(registration())
                    if side == "source":
                        m.kv_args.draft_kv_token_major = False
                    else:
                        info.dst_draft_token_major = False
                    with self.assertRaisesRegex(ValueError, "token-major"):
                        m.prepare_dcp_entry_geometry(info)

    def test_metadata_real_pool_buffers(self):
        for layout in ["nhd", "vectorized_5d"]:
            with self.subTest(layout=layout):
                p = pool(layout)
                a = KVArgs()  # Production publishes target page_size after metadata.
                set_draft_kv_transfer_metadata(
                    a, p, draft_kv_entry_count=2, tp_size=16, total_head_num=16
                )
                self.assertEqual(a.draft_kv_entry_count, 2)
                self.assertEqual(
                    getattr(a, "draft_kv_token_major", True), layout == "nhd"
                )

    def test_packed_tail_repro_equal_tp(self):
        import ctypes

        for count in (1, 65):
            with self.subTest(count=count):
                src = pool("vectorized_5d", heads=2)
                dst = pool("vectorized_5d", heads=2)
                for t in src.k_buffer + src.v_buffer:
                    t.fill_(7)
                for t in dst.k_buffer + dst.v_buffer:
                    t.fill_(-1)
                target = torch.zeros(192, dtype=torch.uint8)
                source_target = target.clone()
                m = manager(tp=8)
                sp, sl, si = src.get_contiguous_buf_infos()
                dp, dl, di = dst.get_contiguous_buf_infos()
                m.kv_args.kv_data_ptrs = [source_target.data_ptr()] + sp
                m.kv_args.kv_item_lens = [64] + si
                set_draft_kv_transfer_metadata(
                    m.kv_args, src, draft_kv_entry_count=2, tp_size=8, total_head_num=16
                )
                da = KVArgs()
                da.page_size = 64
                set_draft_kv_transfer_metadata(
                    da, dst, draft_kv_entry_count=2, tp_size=8, total_head_num=16
                )
                wire = registration()
                wire[4] = struct.pack("3Q", target.data_ptr(), *dp)
                wire[18] = struct.pack("3Q", 64, *di)
                wire[23] = b"1" if getattr(da, "draft_kv_token_major", True) else b"0"
                reg = mc.KVArgsRegisterInfo.from_zmq(wire)
                try:
                    m.prepare_dcp_entry_geometry(reg)
                except ValueError as exc:
                    self.assertIn("token-major", str(exc))
                    continue
                m._validate_envelope_kv_layout = lambda *a: None

                def transfer(session, blocks):
                    for a, b, n in blocks:
                        ctypes.memmove(b, a, n)
                    return 0

                m._transfer_data = transfer
                m.send_kvcache_mixed(
                    "s",
                    np.array([0, 1]),
                    np.array([1, 2]),
                    info=reg,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=count,
                )
                missing = 0
                for token in range(count):
                    slot = token + 64
                    missing += int(
                        (
                            dst.v_buffer[0][slot // 64, :, slot % 64 // 8, :, slot % 8]
                            != 7
                        ).sum()
                    )
                self.fail(
                    f"Packed layout accepted: {missing} required V elements missing for {count} tokens at equal TP8"
                )

    def test_real_pool_transfer_heads_and_tail(self):
        import ctypes

        for plain in (True, False):
            for src_tp in (8, 16):
                for count in (1, 65):
                    for dst_rank in range(8):
                        with self.subTest(
                            plain=plain, src_tp=src_tp, count=count, dst_rank=dst_rank
                        ):
                            dst = pool("nhd", heads=2)
                            for t in dst.k_buffer + dst.v_buffer:
                                t.fill_(-1)
                            target = torch.zeros(192, dtype=torch.uint8)
                            receiver = manager()
                            receiver.attn_tp_size = 8
                            receiver.kv_args.engine_rank = dst_rank
                            receiver.is_mla_backend = plain
                            receiver.is_hybrid_mla_backend = not plain
                            info = PrefillServerInfo(src_tp, 1, 1, 1, 64, None, True)
                            info.draft_kv_entry_count = 2
                            receiver._resolve_rank_mapping(info)
                            writes = set()
                            for src_rank in info.target_tp_ranks:
                                src = pool("nhd", heads=16 // src_tp)
                                for t in src.k_buffer + src.v_buffer:
                                    for h in range(16 // src_tp):
                                        t[:, h, :] = (
                                            torch.arange(192)[:, None]
                                            + (src_rank * (16 // src_tp) + h) * 100
                                        )
                                source_target = torch.full((192,), 7, dtype=torch.uint8)
                                m = manager(tp=src_tp)
                                m.is_mla_backend = plain
                                m.is_hybrid_mla_backend = not plain
                                m.attn_tp_rank = src_rank
                                m.kv_args.engine_rank = src_rank
                                ptrs, lens, items = src.get_contiguous_buf_infos()
                                m.kv_args.kv_data_ptrs = [
                                    source_target.data_ptr()
                                ] + ptrs
                                m.kv_args.kv_item_lens = [64] + items
                                set_draft_kv_transfer_metadata(
                                    m.kv_args,
                                    src,
                                    draft_kv_entry_count=2,
                                    tp_size=src_tp,
                                    total_head_num=16,
                                )
                                dp, dl, di = dst.get_contiguous_buf_infos()
                                da = KVArgs()
                                da.page_size = 64
                                set_draft_kv_transfer_metadata(
                                    da,
                                    dst,
                                    draft_kv_entry_count=2,
                                    tp_size=8,
                                    total_head_num=16,
                                )
                                wire = registration()
                                wire[4] = struct.pack("3Q", target.data_ptr(), *dp)
                                wire[7] = str(dst_rank).encode()
                                wire[18] = struct.pack("3Q", 64, *di)
                                wire[23] = (
                                    b"1"
                                    if getattr(da, "draft_kv_token_major", True)
                                    else b"0"
                                )
                                reg = mc.KVArgsRegisterInfo.from_zmq(wire)
                                m.prepare_dcp_entry_geometry(reg)
                                m._validate_envelope_kv_layout = lambda *a: None

                                def transfer(session, blocks):
                                    for a, b, n in blocks:
                                        for address in range(b, b + n):
                                            self.assertNotIn(address, writes)
                                            writes.add(address)
                                        ctypes.memmove(b, a, n)
                                    return 0

                                m._transfer_data = transfer
                                self.assertEqual(
                                    m.send_kvcache_mixed(
                                        "s",
                                        np.array([0, 1]),
                                        np.array([1, 2]),
                                        info=reg,
                                        src_page_offset=0,
                                        decode_prefix_len=0,
                                        num_kv_tokens=count,
                                    ),
                                    0,
                                )
                            self.assertTrue(torch.all(target[64 : 64 + count] == 7))
                            self.assertTrue(torch.all(target[64 + count :] == 0))
                            for t in dst.k_buffer + dst.v_buffer:
                                for h in range(2):
                                    torch.testing.assert_close(
                                        t[64 : 64 + count, h, :],
                                        (
                                            torch.arange(count)[:, None]
                                            + (2 * dst_rank + h) * 100
                                        )
                                        .expand(count, 8)
                                        .half(),
                                        rtol=0,
                                        atol=0,
                                    )
                                self.assertTrue(torch.all(t[64 + count :] == -1))
                            self.assertEqual(len(writes), count * (1 + 2 * 2 * 8 * 2))

    def test_mha_not_rejected_for_packed_metadata(self):
        m = manager()
        m.is_mla_backend = False
        m.kv_args.draft_kv_token_major = False
        info = mc.KVArgsRegisterInfo.from_zmq(registration(token_major=False))
        info.requires_dcp_relayout = False
        self.assertFalse(m._requires_dcp_entry_geometry(info))


if __name__ == "__main__":
    unittest.main(verbosity=2)
