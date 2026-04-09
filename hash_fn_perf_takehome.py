"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
from turtle import st
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, list | tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            if isinstance(slot, list):
                instrs.append({engine: slot})
            else:
                instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    # def hash_op_parallel(self, op, dest, src1, src2, round, st, end):
    #     slots = []
    #     for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
    #     return ("valu", slots)

    def build_hash_parallel(self, val_hash_addrs, tmp1_parallel, tmp2_parallel, fixed1_parallel, fixed2_parallel, round, st, end):
        instrs = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                
            instrs.append(("debug", [("compare", val_hash_addrs + i, (round, i + st, "pre_hash_stage", hi)) for i in range(0, end - st)]))

            slots = [("vbroadcast", fixed1_parallel, self.scratch_const(val1)), ("vbroadcast", fixed2_parallel, self.scratch_const(val3))]
            instrs.append(("valu", slots))

            # op1
            for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                slots = [(op1, tmp1_parallel + j, val_hash_addrs + j, fixed1_parallel) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                instrs.append(("valu", slots))

            instrs.append(("debug", [("compare", tmp1_parallel + i, (round, i + st, "hash_stage1", hi)) for i in range(0, end - st)]))

            # op3
            for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                slots = [(op3, tmp2_parallel + j, val_hash_addrs + j, fixed2_parallel) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                instrs.append(("valu", slots))

            instrs.append(("debug", [("compare", tmp2_parallel + i, (round, i + st, "hash_stage2", hi)) for i in range(0, end - st)]))

            # op2
            for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                slots = [(op2, val_hash_addrs + j, tmp1_parallel + j, tmp2_parallel + j) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                instrs.append(("valu", slots))

            instrs.append(("debug", [("compare", val_hash_addrs + i, (round, i + st, "hash_stage", hi)) for i in range(0, end - st)]))

        return instrs
    
    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "pre_hash_stage", hi))))
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots
    
    def build_idx_wrap(self, scratch_inp_idx, chunk_len, zero_const_vlen):
        instrs = []
        for i in range(0, chunk_len, SLOT_LIMITS["valu"] * VLEN):
            slots = [("vbroadcast", scratch_inp_idx + j, zero_const_vlen) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, chunk_len), VLEN)]
            instrs.append(("valu", slots))
        return instrs
    
    def build_idx_next(self, scratch_inp_idx, scratch_inp_val, tmp1_parallel, chunk_len, one_const_vlen, two_const_vlen):
        instrs = []

        # tmp1 = val % 2
        for i in range(0, chunk_len, SLOT_LIMITS["valu"] * VLEN):
            slots = [("%", tmp1_parallel + j, scratch_inp_val + j, two_const_vlen) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, chunk_len), VLEN)]
            instrs.append(("valu", slots))

        # tmp1 = tmp1 + 1
        for i in range(0, chunk_len, SLOT_LIMITS["valu"] * VLEN):
            slots = [("+", tmp1_parallel + j, tmp1_parallel + j, one_const_vlen) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, chunk_len), VLEN)]
            instrs.append(("valu", slots))

        # val = (val * 2) + tmp1
        for i in range(0, chunk_len, SLOT_LIMITS["valu"] * VLEN):
            slots = [("multiply_add", scratch_inp_idx + j, scratch_inp_idx + j, two_const_vlen, tmp1_parallel + j) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, chunk_len), VLEN)]
            instrs.append(("valu", slots))

        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Load inputs and forest values into memory to avoid duplicate loads/stores
        inp_indices = self.alloc_scratch("inp_indices", length=batch_size)
        inp_values = self.alloc_scratch("inp_values", length=batch_size)

        inp_val_offsets = [self.alloc_scratch(f"inp_val_offset{o1}") for o1 in range(0, batch_size, VLEN)]
        n_val_offsets = len(inp_val_offsets)

        # initialize the offsets with the beginning of the input values
        for i in range(0, n_val_offsets, SLOT_LIMITS["load"]):
            slots = [("const", inp_val_offsets[j], j * VLEN) for j in range(i, min(i + SLOT_LIMITS["load"], n_val_offsets))]
            body.append(("load", slots))

        # generate offsets in mem from which to vload input values
        for i in range(0, n_val_offsets, SLOT_LIMITS["alu"]):
            slots = [("+", inp_val_offsets[j], inp_val_offsets[j], self.scratch["inp_values_p"]) for j in range(i, min(i + SLOT_LIMITS["alu"], n_val_offsets))]
            body.append(("alu", slots))

        for i in range(0, len(inp_val_offsets), SLOT_LIMITS["load"]):
            slots = [("vload", inp_values + i * VLEN, inp_val_offsets[i])]
            ni = i + 1
            if ni < len(inp_val_offsets):  # handle case where batch_size is not a multiple of VLEN
                slots.append(("vload", inp_values + ni * VLEN, inp_val_offsets[ni]))
            body.append(("load", slots))


        # computation buffers
        parallel_vals = 48 
        assert parallel_vals < SLOT_LIMITS["debug"] , "Parallel vals must be less than debug slot limit to avoid overflowing debug info"

        node_vals = self.alloc_scratch("node_vals", length=parallel_vals)
        tmp_addrs = self.alloc_scratch("tmp_addrs", length=parallel_vals)
        tmp1_parallel = self.alloc_scratch("tmp1_parallel", length=parallel_vals)
        tmp2_parallel = self.alloc_scratch("tmp2_parallel", length=parallel_vals)

        # scratch to support SIMD operations with constants
        fixed1_parallel = self.alloc_scratch("fixed_val_parallel", length=VLEN)
        fixed2_parallel = self.alloc_scratch("fixed_val_parallel", length=VLEN)
        fixed3_parallel = self.alloc_scratch("fixed_val_parallel", length=VLEN)
        # tmp3_parallel = self.alloc_scratch("tmp3_parallel", length=parallel_vals)

        # use vbroadcast to 
        for round in range(rounds):

            # parallel path: take parallel_vals chunks of batch size and process
            for st in range(0, batch_size, parallel_vals):

                end = min(st + parallel_vals, batch_size)

                scratch_inp_idx = inp_indices + st
                scratch_inp_val = inp_values + st

                # check input indices / values indexed in full batch
                body.append(("debug", [("compare", inp_indices + i, (round, i, "idx")) for i in range(st,end)]))
                body.append(("debug", [("compare", inp_values + i, (round, i, "val")) for i in range(st,end)]))

                # broadcast forest location in mem
                for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                    slots = [("vbroadcast", tmp_addrs + j, self.scratch["forest_values_p"]) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                    body.append(("valu", slots))

                # calculate node indices in tmp_addrs
                for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                    slots = [("+", tmp_addrs + j, tmp_addrs + j, scratch_inp_idx + j) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                    body.append(("valu", slots))

                # load node values in node_vals
                for i in range(0, end - st, SLOT_LIMITS["load"]):
                    slots = [("load", node_vals + j, tmp_addrs + j) for j in range(i, min(i + SLOT_LIMITS["load"], end - st), )]
                    body.append(("load", slots))

                # check node values indexed in mini (parallel) batch
                body.append(("debug", [("compare", node_vals + (i - st), (round, i, "node_val")) for i in range(st, end)]))

                # perform XOR with node values in parallel
                for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                    slots = [("^", scratch_inp_val + j, scratch_inp_val + j, node_vals + j) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                    body.append(("valu", slots))

                body.extend(self.build_hash_parallel(scratch_inp_val, tmp1_parallel, tmp2_parallel, fixed1_parallel, fixed2_parallel, round, st, end))
                body.append(("debug", [("compare", scratch_inp_val + (i - st), (round, i, "hashed_val")) for i in range(st, end)]))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                slots = [("vbroadcast", fixed1_parallel, one_const) , ("vbroadcast", fixed2_parallel, two_const), ("vbroadcast", fixed3_parallel, zero_const)]
                body.append(("valu", slots)) # can pack more into all these..

                # if at full depth, set idx to 0
                if (round + 1) % (forest_height + 1) == 0:
                    body.extend(self.build_idx_wrap(scratch_inp_idx, end - st, fixed3_parallel))
                else:
                    body.extend(self.build_idx_next(scratch_inp_idx, scratch_inp_val, tmp1_parallel, end - st, fixed1_parallel, fixed2_parallel))

                body.append(("debug", [("compare", scratch_inp_idx + (i - st), (round, i, "wrapped_idx")) for i in range(st, end)]))


        # use vstore operations to write the inputs back to memory
        for i in range(0, n_val_offsets , SLOT_LIMITS["store"]):
            slots = [("vstore", inp_val_offsets[j], inp_values + j * VLEN) for j in range(i, min(i + SLOT_LIMITS["store"], n_val_offsets))]
            body.append(("store", slots))

        print("Total scratch used: ", self.scratch_ptr, "remaining: ", SCRATCH_SIZE - self.scratch_ptr)
        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_print(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=False, prints=True)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
