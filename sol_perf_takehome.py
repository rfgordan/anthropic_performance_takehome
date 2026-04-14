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

    def build_hash_opt(self, body, inp_val_instr_idxs, val_hash_addrs, tmp1_parallel, hash_consts_vlen, round, st, end):

        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                
            if hi != 3:
                for i in range(end - st):
                    self.interleave_engine_fns(body, ("debug", [("compare", val_hash_addrs + i, (round, st + i, "pre_hash_stage", hi))]), inp_val_instr_idxs[i // VLEN])

            val1_const_vlen, val3_const_vlen = hash_consts_vlen[hi]
            after_hash_instr_idxs = []
            # print(inp_val_instr_idxs)

            # for stage 2, we do two multiply_adds, then xor in stage 3
            if hi == 2:
                next_val1_const_vlen, next_val3_const_vlen = hash_consts_vlen[hi+1]
                for i in range(0, end - st, VLEN):
                    slots = [("multiply_add", tmp1_parallel + i, val_hash_addrs + i, val3_const_vlen, val1_const_vlen)]
                    # idx = i // (VLEN)
                    # last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[idx])
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    # after_hash_instr_idxs.append(last_hash)

                # inp_val_instr_idxs = after_hash_instr_idxs
                # after_hash_instr_idxs = []

                for i in range(0, end - st, VLEN):
                    slots = [("multiply_add", val_hash_addrs + i, val_hash_addrs + i, next_val3_const_vlen, next_val1_const_vlen)]
                    # idx = i // (VLEN)
                    # last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[idx])
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    after_hash_instr_idxs.append(last_hash)
            elif hi == 3:
                # print(inp_val_instr_idxs[idx])
                for i in range(0, end - st, VLEN):
                    slots = [("^", val_hash_addrs + i, tmp1_parallel + i, val_hash_addrs + i)]
                    # idx = i // (VLEN)
                    # last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[idx])
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    after_hash_instr_idxs.append(last_hash)
            # merged multiply_add
            elif op3 == "<<" and op2 == "+" and op1 == "+":
                for i in range(0, end - st, VLEN):
                    slots = [("multiply_add", val_hash_addrs + i, val_hash_addrs + i, val3_const_vlen, val1_const_vlen)]
                    # idx = i // (VLEN)
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    after_hash_instr_idxs.append(last_hash)
            # default path
            else:
                # op1
                for i in range(0, end - st, VLEN):
                    slots = [(op1, tmp1_parallel + i, val_hash_addrs + i, val1_const_vlen)]
                    # idx = i // (VLEN)
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    after_hash_instr_idxs.append(last_hash)

                # instrs.append(("debug", [("compare", tmp1_parallel + i, (round, st + i, "hash_stage1", hi)) for i in range(0, end - st)]))
                inp_val_instr_idxs = after_hash_instr_idxs
                after_hash_instr_idxs = []
                # print(inp_val_instr_idxs)

                # op3
                for i in range(0, end - st, VLEN):
                    slots = [(op3, val_hash_addrs + i, val_hash_addrs + i, val3_const_vlen)]
                    # idx = i // (VLEN)
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    after_hash_instr_idxs.append(last_hash)

                # instrs.append(("debug", [("compare", tmp2_parallel + i, (round, st + i, "hash_stage2", hi)) for i in range(0, end - st)]))

                inp_val_instr_idxs = after_hash_instr_idxs
                after_hash_instr_idxs = []
                # print(inp_val_instr_idxs)

                # op2
                for i in range(0, end - st, VLEN):
                    slots = [(op2, val_hash_addrs + i, tmp1_parallel + i, val_hash_addrs + i)]
                    # idx = i // (VLEN)
                    last_hash = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
                    after_hash_instr_idxs.append(last_hash)

            inp_val_instr_idxs = after_hash_instr_idxs
            if hi != 2:
                for i in range(end - st):
                    self.interleave_engine_fns(body, ("debug", [("compare", val_hash_addrs + i, (round, st + i, "hash_stage", hi))]), inp_val_instr_idxs[i // VLEN])

        return inp_val_instr_idxs

    def build_hash_parallel(self, val_hash_addrs, tmp1_parallel, tmp2_parallel, fixed1_parallel, fixed2_parallel, round, st, end):
        instrs = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                
            instrs.append(("debug", [("compare", val_hash_addrs + i, (round, st + i, "pre_hash_stage", hi)) for i in range(0, end - st)]))

            slots = [("vbroadcast", fixed1_parallel, self.scratch_const(val1)), ("vbroadcast", fixed2_parallel, self.scratch_const(val3))]
            instrs.append(("valu", slots))

            # op1
            for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                slots = [(op1, tmp1_parallel + j, val_hash_addrs + j, fixed1_parallel) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                instrs.append(("valu", slots))

            instrs.append(("debug", [("compare", tmp1_parallel + i, (round, st + i, "hash_stage1", hi)) for i in range(0, end - st)]))

            # op3
            for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                slots = [(op3, tmp2_parallel + j, val_hash_addrs + j, fixed2_parallel) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                instrs.append(("valu", slots))

            instrs.append(("debug", [("compare", tmp2_parallel + i, (round, st + i, "hash_stage2", hi)) for i in range(0, end - st)]))

            # op2
            for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
                slots = [(op2, val_hash_addrs + j, tmp1_parallel + j, tmp2_parallel + j) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
                instrs.append(("valu", slots))

            instrs.append(("debug", [("compare", val_hash_addrs + i, (round, st + i, "hash_stage", hi)) for i in range(0, end - st)]))

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
    
    def build_idx_wrap(self, body, inp_idx_instr_idxs, scratch_inp_idx, chunk_len, zero_const_vlen):

        for i in range(0, chunk_len, VLEN):
            slots = [("vbroadcast", scratch_inp_idx + i, zero_const_vlen)]
            inp_idx_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_idx_instr_idxs[i // VLEN])

        return inp_idx_instr_idxs
    
    def build_idx_next(self, body, inp_val_instr_idxs, scratch_inp_idx, scratch_inp_val, tmp1_parallel, tmp2_parallel, chunk_len, one_const_vlen, two_const_vlen):

        # print(inp_val_instr_idxs)

        # tmp1 = val % 2
        for i in range(0, chunk_len, VLEN):
            slots = [("%", tmp1_parallel + i, scratch_inp_val + i, two_const_vlen)]
            inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # print(inp_val_instr_idxs)

        # tmp1 = tmp1 + 1
        for i in range(0, chunk_len, VLEN):
            slots = [("+", tmp1_parallel + i, tmp1_parallel + i, one_const_vlen)]
            inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # print(inp_val_instr_idxs)

        # idx = (idx * 2) + tmp1
        for i in range(0, chunk_len, VLEN):
            slots = [("multiply_add", scratch_inp_idx + i, scratch_inp_idx + i, two_const_vlen, tmp1_parallel + i)]
            inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        return inp_val_instr_idxs
    
    def build_apply_node_val_root(self, body, inp_val_instr_idxs, inp_values, root_node_val_vlen, chunk_len):

        next_instr_idxs = []

        for i in range(0, chunk_len, VLEN):
            slots = [("^", inp_values + i, inp_values + i, root_node_val_vlen)]
            last_xor = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
            next_instr_idxs.append(last_xor)

        # return inp_val_instr_idxs, [curr_instr] * len(inp_val_instr_idxs)
        return next_instr_idxs


    def build_apply_node_val_masked(self, body, inp_val_instr_idxs, inp_values, inp_indices, node_vals, tmp1_parallel, tree_vals_vlen, consts_vlen, round, depth, chunk_len):
        
        next_instr_idxs = []

        # set node_vals to 0
        if round > 0:
            for i in range(0, chunk_len, VLEN):
                slots = [("vbroadcast", node_vals + i, consts_vlen[0])]
                self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # iterate over all possible tree nodes
        for i in range(2**depth - 1, 2**(depth + 1) - 1):

            post_mask_instr_idxs = []

            # mask input indices vs constants
            for j in range(0, chunk_len, VLEN):
                slots = [("==", tmp1_parallel + j, inp_indices + j, consts_vlen[i])]
                post_mask_instr_idx = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[j // VLEN])
                post_mask_instr_idxs.append(post_mask_instr_idx)

            # add node value if mask is true
            for j in range(0, chunk_len, VLEN):
                slots = [("multiply_add", node_vals + j, tmp1_parallel + j, tree_vals_vlen[i], node_vals + j)]

                # we want to depend on the final m_add
                next_instr_idx = self.interleave_engine_fns(body, ("valu", slots), post_mask_instr_idxs[j // VLEN])
                next_instr_idxs.append(next_instr_idx)

            inp_val_instr_idxs = next_instr_idxs
            next_instr_idxs = []

        for i in range(0, chunk_len, VLEN):
            slots = [("^", inp_values + i, node_vals + i, inp_values + i)]
            last_xor = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
            next_instr_idxs.append(last_xor)

        return next_instr_idxs
    
    def build_apply_node_val_scratch(self, body, scratch_inp_idx, scratch_inp_val, scratch_node_val, round, st, end):
        return []
    
    def build_apply_node_val_mem(self, body, inp_val_instr_idxs, inp_indices, inp_values, node_vals, forest_p_const_vlen, round, st, end):
        
        # curr_last_dep = len(body)
        last_loads = []
        last_broadcasts = []

        # broadcast forest location in mem
        # for i in range(0, end - st, SLOT_LIMITS["valu"] * VLEN):
        #     slots = [("+", inp_indices + j, inp_indices + j, forest_p_const_vlen) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, end - st), VLEN)]
        #     last_broadcast = self.interleave_engine_fns(body, ("valu", slots), curr_last_dep)

        for i in range(0, end - st, VLEN):
            slots = [("+", inp_indices + i, inp_indices + i, forest_p_const_vlen)]
            last_broadcast = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
            last_broadcasts.append(last_broadcast)

        # load node values in node_vals
        for i in range(0, end - st):
            slots = [("load", node_vals + i, inp_indices + i)]
            last_load = self.interleave_engine_fns(body, ("load", slots), last_broadcasts[i // VLEN])
            last_loads.append(last_load)

        # check node values indexed in mini (parallel) batch
        self.interleave_engine_fns(body, ("debug", [("compare", node_vals + i, (round, st + i, "node_val")) for i in range(0, end - st)]), last_loads[i])

        # perform XOR with node values in parallel
        for i in range(0, end - st, VLEN):
            slots = [("^", inp_values + i, inp_values + i, node_vals + i)]
            self.interleave_engine_fns(body, ("valu", slots), max(last_loads[i:i+VLEN]))
            # inp_val_instr_idxs.append(last_xor)

        # broadcast forest location in mem
        inp_val_instr_idxs = []
        for i in range(0, end - st, VLEN):
            slots = [("-", inp_indices + i, inp_indices + i, forest_p_const_vlen)]
            last_minus = self.interleave_engine_fns(body, ("valu", slots), max(last_loads[i:i+VLEN]))
            inp_val_instr_idxs.append(last_minus)

        return inp_val_instr_idxs
    
    # after this call, should have vectors for first six indices / values
    def build_load_tree_vals(self, body, tree_vals_vlen, consts_vlen):

        for i in range(0, len(tree_vals_vlen), SLOT_LIMITS["alu"]):
            self.interleave_engine_fns(body, ("alu", [("+", tree_vals_vlen[j], self.scratch["forest_values_p"], consts_vlen[j]) for j in range(i, min(i + SLOT_LIMITS["alu"], len(tree_vals_vlen)))]))

        for i, tree_val_vlen in enumerate(tree_vals_vlen):
            slots = [("load", tree_val_vlen, tree_val_vlen) for _ in range(i, min(i + SLOT_LIMITS["load"], len(tree_vals_vlen)))]
            self.interleave_engine_fns(body, ("load", slots))

        slots = []
        for i, tree_val_vlen in enumerate(tree_vals_vlen):
            slots.append(("vbroadcast", tree_val_vlen, tree_val_vlen))

        for i in range(0, len(slots), SLOT_LIMITS["valu"]):
            self.interleave_engine_fns(body, ("valu", slots[i:min(i+SLOT_LIMITS["valu"], len(slots))]))
    
    @staticmethod
    def interleave_engine_fns(body, slots, first_possible=None):
        engine, slots = slots
        slots = slots if isinstance(slots, list) else [slots]
        first_possible = len(body) if first_possible is None else first_possible
        
        for i in range(first_possible, len(body)):
            instr = body[i]
            if engine not in instr:
                instr[engine] = slots
                return i + 1

            if engine in instr:
                if len(instr[engine]) + len(slots) < SLOT_LIMITS[engine]:
                    instr[engine].extend(slots)
                    return i + 1
                
                remaining_slots = SLOT_LIMITS[engine] - len(instr[engine])
                instr[engine].extend(slots[:remaining_slots])
                slots = slots[remaining_slots:]

        body.append({engine: slots})
        return len(body)
                
    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

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

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        n_tree_preload_layers = 4
        n_tree_preload_layers = min(n_tree_preload_layers, forest_height + 1) # can't preload more layers than the tree has
        consts_vlen = [self.alloc_scratch(f"const_{val}_vlen", length=VLEN) for val in range(2**n_tree_preload_layers - 1)]
        tree_vals_vlen = [self.alloc_scratch(f"tree_val_{i}_vlen", length=VLEN) for i in range(2**n_tree_preload_layers - 1)]

        forest_p_const_vlen = self.alloc_scratch("forest_p_const_vlen", length=VLEN)
        const_slots = [("vbroadcast", consts_vlen[0], zero_const) , ("vbroadcast", consts_vlen[1], one_const), ("vbroadcast", consts_vlen[2], two_const), ("vbroadcast", forest_p_const_vlen, self.scratch["forest_values_p"])]

        hash_consts_vlen = []
        for hi, (_, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op2 == "+" and op3 == "<<":
                # if combining instructions, we need to make the constant 2 ^ val3
                val3 = 2 ** val3 + 1
            # special handling for stage 3: do two multiply_adds, then xor in stage 4
            if hi == 2:
                val1 += HASH_STAGES[3][1]
            if hi == 3:
                prev_val1 = HASH_STAGES[2][1]
                prev_val3 = HASH_STAGES[2][4]
                val1 = prev_val1 * (2 ** val3)
                val3 = (2 ** prev_val3 + 1) * (2 ** val3)
            hash_const1_vlen = self.alloc_scratch(f"hash_const1_vlen_{hi}_{val1}", length=VLEN)
            hash_const3_vlen = self.alloc_scratch(f"hash_const3_vlen_{hi}_{val3}", length=VLEN)
            hash_consts_vlen.append((hash_const1_vlen, hash_const3_vlen))
            const_slots.append(("vbroadcast", hash_const1_vlen, self.scratch_const(val1)))
            const_slots.append(("vbroadcast", hash_const3_vlen, self.scratch_const(val3)))

        for i in range(0, len(const_slots), SLOT_LIMITS["valu"]):
            slots = const_slots[i:min(i+SLOT_LIMITS["valu"], len(const_slots))]
            self.interleave_engine_fns(body, ("valu", slots))
            # body.append(("valu", slots))

        slots = [("+", consts_vlen[3], consts_vlen[1], consts_vlen[2]), ("*", consts_vlen[4], consts_vlen[2], consts_vlen[2]), ("multiply_add", consts_vlen[5], consts_vlen[2], consts_vlen[2], consts_vlen[1]), ("multiply_add", consts_vlen[6], consts_vlen[2], consts_vlen[2], consts_vlen[2])]
        self.interleave_engine_fns(body, ("valu", slots))

        if len(consts_vlen) > 7:
            for i in range(7, len(consts_vlen)):
                slots = [("const", consts_vlen[j], j) for j in range(i, min(i + SLOT_LIMITS["load"], len(consts_vlen)))]
                self.interleave_engine_fns(body, ("load", slots))

            for i in range(7, len(consts_vlen), SLOT_LIMITS["valu"]):
                slots = [("vbroadcast", consts_vlen[j], consts_vlen[j]) for j in range(i, min(i + SLOT_LIMITS["valu"], len(consts_vlen)))]
                self.interleave_engine_fns(body, ("valu", slots))

        # computation buffers
        parallel_vals = 48 
        assert parallel_vals < SLOT_LIMITS["debug"] , "Parallel vals must be less than debug slot limit to avoid overflowing debug info"
        chunk_incr = self.scratch_const(parallel_vals, "chunk_incr")

        n_val_offsets = parallel_vals // VLEN # 6
        inp_val_offsets = self.alloc_scratch("inp_val_offsets", length=n_val_offsets)

        node_vals = self.alloc_scratch("node_vals", length=parallel_vals)
        # tmp_addrs = self.alloc_scratch("tmp_addrs", length=parallel_vals)
        tmp1_parallel = self.alloc_scratch("tmp1_parallel", length=parallel_vals)
        tmp2_parallel = self.alloc_scratch("tmp2_parallel", length=parallel_vals)

        # n_layers_in_mem_forest = 0
        self.build_load_tree_vals(body, tree_vals_vlen, consts_vlen)

        # Load inputs and forest values into memory to avoid duplicate loads/stores
        inp_indices = self.alloc_scratch("inp_indices", length=parallel_vals)
        inp_values = self.alloc_scratch("inp_values", length=parallel_vals)

        # initialize the offsets with the beginning of the input values
        for i in range(0, n_val_offsets, SLOT_LIMITS["load"]):
            slots = [("const", inp_val_offsets + j, j * VLEN) for j in range(i, min(i + SLOT_LIMITS["load"], n_val_offsets))]
            self.interleave_engine_fns(body, ("load", slots))

        # generate offsets in mem from which to vload input values
        for i in range(0, n_val_offsets, SLOT_LIMITS["alu"]):
            slots = [("+", inp_val_offsets + j, inp_val_offsets + j, self.scratch["inp_values_p"]) for j in range(i, min(i + SLOT_LIMITS["alu"], n_val_offsets))]
            self.interleave_engine_fns(body, ("alu", slots))

        # parallel path: take parallel_vals chunks of batch size and process
        for ci, st in enumerate(range(0, batch_size, parallel_vals)):

            end = min(st + parallel_vals, batch_size)
            chunk_len = end - st

            curr_instr_len = len(body)

            if ci > 0:
                # if not the first chunk, reset index values
                for i in range(0, parallel_vals, SLOT_LIMITS["valu"] * VLEN):
                    slots = [("vbroadcast", inp_indices + j, consts_vlen[0]) for j in range(i, min(i + SLOT_LIMITS["valu"] * VLEN, parallel_vals), VLEN)]
                    self.interleave_engine_fns(body, ("valu", slots), curr_instr_len)

                # increment offsets by parallel_vals for next chunk
                for i in range(0, n_val_offsets, SLOT_LIMITS["alu"]):
                    slots = [("+", inp_val_offsets + j, inp_val_offsets + j, chunk_incr) for j in range(i, min(i + SLOT_LIMITS["alu"], n_val_offsets))]
                    self.interleave_engine_fns(body, ("alu", slots), curr_instr_len)

            curr_instr_len = len(body)

            assert chunk_len % VLEN == 0, "If chunk length isn't a multiple of VLEN, vload could overrun inp_values"
            n_val_offsets = min(n_val_offsets, chunk_len // VLEN + int(chunk_len % VLEN != 0))
            for i in range(0, n_val_offsets, SLOT_LIMITS["load"]):
                slots = [("vload", inp_values + i * VLEN, inp_val_offsets + i)]
                ni = i + 1
                if ni < n_val_offsets:  # handle case where batch_size is not a multiple of VLEN
                    slots.append(("vload", inp_values + ni * VLEN, inp_val_offsets + ni))
                self.interleave_engine_fns(body, ("load", slots), curr_instr_len)

            # larger than necessary
            inp_val_instr_idxs = [len(body)] * parallel_vals

            # use vbroadcast to 
            for round in range(rounds):
                depth = round % (forest_height + 1)
                
                # inp_val_instr_idxs in VLEN strided format

                # check input indices / values indexed in full batch
                for i in range(0,end - st):
                    self.interleave_engine_fns(body, ("debug", [("compare", inp_indices + i, (round, st + i, "idx"))]), inp_val_instr_idxs[i // VLEN])
                    self.interleave_engine_fns(body, ("debug", [("compare", inp_values + i, (round, st + i, "val"))]), inp_val_instr_idxs[i // VLEN])

                if depth == 0:
                    inp_val_instr_idxs = self.build_apply_node_val_root(body, inp_val_instr_idxs, inp_values, tree_vals_vlen[0], chunk_len)
                elif depth < n_tree_preload_layers:
                    # print("Before: ", inp_val_instr_idxs)
                    inp_val_instr_idxs = self.build_apply_node_val_masked(body, inp_val_instr_idxs, inp_values, inp_indices, node_vals, tmp1_parallel, tree_vals_vlen, consts_vlen, round, depth, chunk_len)
                    # print("After: ", inp_val_instr_idxs)
                else:
                    inp_val_instr_idxs = self.build_apply_node_val_mem(body, inp_val_instr_idxs, inp_indices, inp_values, node_vals, forest_p_const_vlen, round, st, end)

                inp_val_instr_idxs = self.build_hash_opt(body, inp_val_instr_idxs, inp_values, tmp1_parallel, hash_consts_vlen, round, st, end)
                for i in range(0, end-st):
                    self.interleave_engine_fns(body,("debug", [("compare", inp_values + i, (round, st + i, "hashed_val"))]), inp_val_instr_idxs[i // VLEN])

                # if at full depth, set idx to 0
                if (round + 1) % (forest_height + 1) == 0:
                    inp_val_instr_idxs = self.build_idx_wrap(body, inp_val_instr_idxs, inp_indices, end - st, consts_vlen[0])
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                else:
                    inp_val_instr_idxs = self.build_idx_next(body, inp_val_instr_idxs, inp_indices, inp_values, tmp1_parallel, tmp2_parallel, end - st, consts_vlen[1], consts_vlen[2])

                for i in range(0, end - st):
                    self.interleave_engine_fns(body,("debug", [("compare", inp_indices + i, (round, st + i, "wrapped_idx"))]), inp_val_instr_idxs[i // VLEN])

                # on last round, can potentially skip index update

            # use vstore operations to write the inputs back to memory
            for i in range(0, min(n_val_offsets, chunk_len // VLEN), SLOT_LIMITS["store"]):
                slots = [("vstore", inp_val_offsets + j, inp_values + j * VLEN) for j in range(i, min(i + SLOT_LIMITS["store"], n_val_offsets))]
                self.interleave_engine_fns(body,("store", slots))

        print("Total scratch used: ", self.scratch_ptr, "remaining: ", SCRATCH_SIZE - self.scratch_ptr)
        
        # print(body)
        self.instrs.extend(body)
        self.print_instructions()
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def print_instructions(self):
        for instr in self.instrs:
            print(instr)

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

    def test_kernel_trace_small(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 3, 256, trace=True, prints=False)

    def test_kernel_print(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=False, prints=True)

    def test_kernel_print_awk_size(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 261, trace=False, prints=True)

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
