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
        slot = (*slot, {})
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

    def build_hash_opt(self, body, i, inp_val_instr_idxs, val_hash_addrs, tmp1_parallel, hash_consts_vlen, round, st, end, debug_info):

        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                
            if hi != 3:
                for j in range(i,i+VLEN):
                    self.interleave_engine_fns(body, ("debug", ("compare", val_hash_addrs + j, (round, st + j, "pre_hash_stage", hi))), inp_val_instr_idxs[i // VLEN])

            val1_const_vlen, val3_const_vlen = hash_consts_vlen[hi]

            # for stage 2, we do two multiply_adds, then xor in stage 3
            if hi == 2:
                next_val1_const_vlen, next_val3_const_vlen = hash_consts_vlen[hi+1]
                slots = ("multiply_add", tmp1_parallel + i, val_hash_addrs + i, val3_const_vlen, val1_const_vlen)
                self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)

                slots = ("multiply_add", val_hash_addrs + i, val_hash_addrs + i, next_val3_const_vlen, next_val1_const_vlen)
                inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)
            elif hi == 3:
                slots = ("^", val_hash_addrs + i, tmp1_parallel + i, val_hash_addrs + i)
                inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)

            # merged multiply_add
            elif op3 == "<<" and op2 == "+" and op1 == "+":
                slots = ("multiply_add", val_hash_addrs + i, val_hash_addrs + i, val3_const_vlen, val1_const_vlen)
                inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)

            # default path
            else:
                # op1
                slots = (op1, tmp1_parallel + i, val_hash_addrs + i, val1_const_vlen)
                self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)

                # instrs.append(("debug", [("compare", tmp1_parallel + i, (round, st + i, "hash_stage1", hi)) for i in range(0, end - st)]))

                # op3
                slots = (op3, val_hash_addrs + i, val_hash_addrs + i, val3_const_vlen)
                inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)

                # instrs.append(("debug", [("compare", tmp2_parallel + i, (round, st + i, "hash_stage2", hi)) for i in range(0, end - st)]))

                # op2
                slots = (op2, val_hash_addrs + i, tmp1_parallel + i, val_hash_addrs + i)
                inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN], debug_info)

            if hi != 2:
                for j in range(i,i+VLEN):
                    self.interleave_engine_fns(body, ("debug", ("compare", val_hash_addrs + j, (round, st + j, "hash_stage", hi))), inp_val_instr_idxs[i // VLEN])

        return inp_val_instr_idxs
    
    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "pre_hash_stage", hi))))
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots
    
    def build_idx_wrap(self, body, i, inp_idx_instr_idxs, scratch_inp_idx, chunk_len, zero_const_vlen):

        slots = ("vbroadcast", scratch_inp_idx + i, zero_const_vlen)
        inp_idx_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_idx_instr_idxs[i // VLEN])

        return inp_idx_instr_idxs
    
    def build_idx_next(self, body, i, inp_val_instr_idxs, scratch_inp_idx, scratch_inp_val, tmp1_parallel, chunk_len, forest_const_m1_vlen, one_const_vlen, two_const_vlen):

        # tmp1 = val % 2
        slots = ("%", tmp1_parallel + i, scratch_inp_val + i, two_const_vlen)
        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # idx = (idx * 2) + tmp1
        slots = ("multiply_add", scratch_inp_idx + i, scratch_inp_idx + i, two_const_vlen, tmp1_parallel + i)
        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # idx = idx - (forest_values_p - 1)
        slots = ("-", scratch_inp_idx + i, scratch_inp_idx + i, forest_const_m1_vlen)
        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        return inp_val_instr_idxs
    
    def build_apply_node_val_root(self, body, i, inp_val_instr_idxs, inp_values, root_node_val_vlen):
        slots = ("^", inp_values + i, inp_values + i, root_node_val_vlen)
        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
        return inp_val_instr_idxs


    def build_apply_node_val_masked(self, body, i, inp_val_instr_idxs, inp_values, inp_indices, node_vals, tmp1_parallel, tree_vals_vlen, tree_idxs_vlen, consts_vlen, round, depth, chunk_len):

        # set node_vals to 0
        if round > 0:
            slots = ("vbroadcast", node_vals + i, consts_vlen[0])
            post_reset_node_vals = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # iterate over all possible tree nodes
        for j in range(2**depth - 1, 2**(depth + 1) - 1):

            # mask input indices vs constants
            slots = ("==", tmp1_parallel + i, inp_indices + i, tree_idxs_vlen[j])
            post_mask_instr_idx = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])
            # add node value if mask is true
            slots = ("multiply_add", node_vals + i, tmp1_parallel + i, tree_vals_vlen[j], node_vals + i)

            # we want to depend on the final m_add
            inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), max(post_mask_instr_idx,post_reset_node_vals))

        slots = ("^", inp_values + i, node_vals + i, inp_values + i)
        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        return inp_val_instr_idxs
    
    def build_apply_node_val_scratch(self, body, scratch_inp_idx, scratch_inp_val, scratch_node_val, round, st, end):
        return []
    
    def build_apply_node_val_mem(self, body, i, inp_val_instr_idxs, inp_indices, inp_values, node_vals, round, st, end, debug_info):
        
        loads = [len(body)] * min(VLEN,end-i)

        # for i in range(0, end - st, VLEN):
        #     slots = [("+", inp_indices + i, inp_indices + i, forest_p_const_vlen)]
        #     inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

        # load node values in node_vals
        for j in range(i,min(i+VLEN,end)):
            slots = ("load", node_vals + j, inp_indices + j)
            loads[j-i] = self.interleave_engine_fns(body, ("load", slots), inp_val_instr_idxs[i // VLEN], debug_info)
            # last_loads.append(last_load)

        # check node values indexed in mini (parallel) batch
            self.interleave_engine_fns(body, ("debug", ("compare", node_vals + i, (round, st + i, "node_val"))), loads[j-i])

        # perform XOR with node values in parallel
        slots = ("^", inp_values + i, inp_values + i, node_vals + i)
        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), max(loads))

        # # broadcast forest location in mem
        # for i in range(0, end - st, VLEN):
        #     slots = [("-", inp_indices + i, inp_indices + i, forest_p_const_vlen)]
        #     inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), last_loads[i // VLEN])

        return inp_val_instr_idxs
    
    # after this call, should have vectors for first six indices / values
    def build_load_tree_vals(self, body, after_init_vars_instr, tree_vals_vlen, consts_vlen):

        next_instr_idxs = [len(body)] * len(tree_vals_vlen)
        for i in range(0, len(tree_vals_vlen)):
            slot = ("+", tree_vals_vlen[i], self.scratch["forest_values_p"], consts_vlen[i])
            next_instr_idxs[i] = self.interleave_engine_fns(body, ("alu", slot), after_init_vars_instr)

        for i, tree_val_vlen in enumerate(tree_vals_vlen):
            slot = ("load", tree_val_vlen, tree_val_vlen)
            next_instr_idxs[i] = self.interleave_engine_fns(body, ("load", slot), next_instr_idxs[i])

        for i, tree_val_vlen in enumerate(tree_vals_vlen):
            slot = ("vbroadcast", tree_val_vlen, tree_val_vlen)
            next_instr_idxs[i] = self.interleave_engine_fns(body, ("valu", slot), next_instr_idxs[i])

        return max(next_instr_idxs)
    
    @staticmethod
    def valu_slot_to_alu_slot(slot):
        op, dest, a1, a2 = slot[:4]
        extra_info = slot[4:]
        # assert core == "valu", "valu_slot_to_alu_slot only takes valu slots"
        slots = [(op, dest + i, a1 + i, a2 + i) + extra_info for i in range(VLEN)]
        return slots


    @staticmethod
    def interleave_engine_fns(body, slot, first_possible=None, extra_info={}, should_pack_valu=True):
        engine, slot = slot
        slot = slot + (extra_info,)
        first_possible = len(body) if first_possible is None else first_possible
        
        for i in range(first_possible, len(body)):
            instr = body[i]
            if engine not in instr:
                instr[engine] = [slot]
                return i + 1
            
            if engine in instr and len(instr[engine]) < SLOT_LIMITS[engine]:
                instr[engine].append(slot)
                return i + 1
            
            if should_pack_valu and engine == "valu" and slot[0] not in ("vbroadcast", "multiply_add") and ("alu" not in instr or len(instr["alu"]) + VLEN <= SLOT_LIMITS["alu"]):
                slots = KernelBuilder.valu_slot_to_alu_slot(slot)
                print("Transformed slots: ", slot, " -> ", slots)
                if "alu" not in instr:
                    instr["alu"] = []

                instr["alu"].extend(slots)
                return i + 1

        body.append({engine: [slot]})
        return len(body)
                
    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """

        # HYPERPARAMETERS
        n_tree_preload_layers = 3
        parallel_vals = 256
        n_val_offsets = parallel_vals // VLEN
        n_tree_preload_layers = min(n_tree_preload_layers, forest_height + 1) # can't preload more layers than the tree has

        # IN-MEMORY HELPERS
        consts_vlen = [self.alloc_scratch(f"const_{val}_vlen", length=VLEN) for val in range(2**n_tree_preload_layers)] # can go back to -1?
        forest_consts_vlen = [self.alloc_scratch(f"forest_const_{val}_vlen", length=VLEN) for val in range(2**n_tree_preload_layers)]
        forest_const_m1_vlen = self.alloc_scratch(f"forest_const_m1_vlen", length=VLEN)
        tree_vals_vlen = [self.alloc_scratch(f"tree_val_{i}_vlen", length=VLEN) for i in range(2**n_tree_preload_layers)]
        inp_val_offsets = self.alloc_scratch("inp_val_offsets", length=n_val_offsets)
        node_vals = self.alloc_scratch("node_vals", length=parallel_vals)
        tmp1_parallel = self.alloc_scratch("tmp1_parallel", length=parallel_vals)
        # Load inputs and forest values into memory to avoid duplicate loads/stores
        inp_indices = self.alloc_scratch("inp_indices", length=parallel_vals)
        inp_values = self.alloc_scratch("inp_values", length=parallel_vals)

        body = []  # array of instructions

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Optimized Loading for Constants [0, 1, 2, 4, 5, 6]
        consts = {0: self.alloc_scratch("const_0")}

        next_instr = None
        init_first_consts = [1,2]
        for c in init_first_consts:
            consts[c] = self.alloc_scratch(f"const_{c}")
            next_instr = self.interleave_engine_fns(body, ("load", ("const", consts[c], c)), 0)
        
        consts[3] = self.alloc_scratch("const_3")
        consts[4] = self.alloc_scratch("const_4")
        self.interleave_engine_fns(body, ("alu", ("+", consts[4], consts[2], consts[2])), next_instr)
        self.interleave_engine_fns(body, ("alu", ("+", consts[3], consts[1], consts[2])), next_instr)

        after_second_consts_instr = None
        init_second_consts = [5,6]
        for c in init_second_consts:
            consts[c] = self.alloc_scratch(f"const_{c}")
            after_second_consts_instr = self.interleave_engine_fns(body, ("load", ("const", consts[c], c)), next_instr)

        init_vlen_consts = [0,1,2]
        for c in init_vlen_consts:
            slot = ("valu", ("vbroadcast", consts_vlen[c], consts[c]))
            self.interleave_engine_fns(body, slot, next_instr)

        after_init_vars_instr = None
        init_vars = {
            "forest_values_p": 4,
            "inp_indices_p": 5,
            "inp_values_p": 6
        }
        for v, c in init_vars.items():
            self.alloc_scratch(v)
            after_init_vars_instr = self.interleave_engine_fns(body, ("load", ("load", self.scratch[v], consts[c])), after_second_consts_instr)

        consts[7] = self.alloc_scratch("const_7")
        after_second_consts_instr = self.interleave_engine_fns(body, ("alu", ("+", consts[7], consts[1], consts[6])), after_second_consts_instr)

        after_vlen_consts_init = None
        second_vlen_consts = [3,4,5,6]
        for c in second_vlen_consts:
            slot = ("valu", ("vbroadcast", consts_vlen[c], consts[c]))
            after_vlen_consts_init = self.interleave_engine_fns(body, slot, after_second_consts_instr)

        # forest_p_const_vlen = self.alloc_scratch("forest_p_const_vlen", length=VLEN)
        slot = ("valu", ("vbroadcast", forest_consts_vlen[0], self.scratch["forest_values_p"]))
        after_forest_vlen_instr = self.interleave_engine_fns(body, slot, after_init_vars_instr)

        slot = ("valu", ("-", forest_const_m1_vlen, forest_consts_vlen[0], consts_vlen[1]))
        self.interleave_engine_fns(body, slot, after_forest_vlen_instr)

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
            val1_const = self.alloc_scratch(f"const_{hi}_{val1}")
            val3_const = self.alloc_scratch(f"const_{hi}_{val3}")
            after_val1_instr = self.interleave_engine_fns(body, ("load", ("const", val1_const, val1)), 0)
            after_val3_instr = self.interleave_engine_fns(body, ("load", ("const", val3_const, val3)), 0)
            self.interleave_engine_fns(body, ("valu", ("vbroadcast", hash_const1_vlen, val1_const)), after_val1_instr)
            self.interleave_engine_fns(body, ("valu", ("vbroadcast", hash_const3_vlen, val3_const)), after_val3_instr)

        if len(consts_vlen) > 7:
            print(f"(!!!!) not optimized path. number of const vectors: {len(consts_vlen)}")

            for i in range(7, len(consts_vlen)):
                slots = ("const", consts_vlen[i], i)
                after_consts_3_init = self.interleave_engine_fns(body, ("load", slots), 0)

                slots = ("vbroadcast", consts_vlen[i], consts_vlen[i])
                after_vlen_consts_init = self.interleave_engine_fns(body, ("valu", slots), after_consts_3_init)

        for i, vc in enumerate(forest_consts_vlen[1:]):
            slot = ("valu", ("+", vc, forest_consts_vlen[0], consts_vlen[i+1]))
            self.interleave_engine_fns(body, slot, after_vlen_consts_init)


        # assert parallel_vals < SLOT_LIMITS["debug"] , "Parallel vals must be less than debug slot limit to avoid overflowing debug info"
        chunk_incr = self.alloc_scratch("chunk_incr")
        after_chunk_incr_idx = self.interleave_engine_fns(body, ("load", ("const", chunk_incr, parallel_vals)), 0)

        after_load_tree_vals_instr = self.build_load_tree_vals(body, max(after_init_vars_instr,after_vlen_consts_init), tree_vals_vlen, consts_vlen)

        # can potentially optimize this using alus
        # initialize the offsets with the beginning of the input values
        after_init_offsets_instrs = [len(body)] * n_val_offsets
        for i in range(0, n_val_offsets):
            slot = ("const", inp_val_offsets + i, i * VLEN)
            after_init_offsets_instrs[i] = self.interleave_engine_fns(body, ("load", slot), 0)

        # generate offsets in mem from which to vload input values
        for i in range(0, n_val_offsets):
            slot = ("+", inp_val_offsets + i, inp_val_offsets + i, self.scratch["inp_values_p"])
            after_init_offsets_instrs[i] = self.interleave_engine_fns(body, ("alu", slot), after_init_offsets_instrs[i])

        # inp_val_instr_idxs = [len(body)] * (parallel_vals // VLEN)
        inp_val_instr_idxs = after_init_offsets_instrs
        # past_offset_instr_idxs = [0] * len(inp_val_instr_idxs)


        # parallel path: take parallel_vals chunks of batch size and process
        for ci, st in enumerate(range(0, batch_size, parallel_vals)):

            end = min(st + parallel_vals, batch_size)
            chunk_len = end - st

            next_instr_idxs = [None] * len(inp_val_instr_idxs)

            assert chunk_len % VLEN == 0, "If chunk length isn't a multiple of VLEN, vload could overrun inp_values"

            def process_vector(round, i, inp_val_instr_idxs):
                debug_info = {"round": round, "st": st, "i": i}
                depth = round % (forest_height + 1)

                if round == 0:
                    if ci > 0:
                        slots = ("+", inp_val_offsets + i // VLEN, inp_val_offsets + i // VLEN, chunk_incr)
                        inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("alu", slots), inp_val_instr_idxs[i // VLEN])

                    slots = ("vbroadcast", inp_indices + i, forest_consts_vlen[0])
                    next_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("valu", slots), inp_val_instr_idxs[i // VLEN])

                    slots = ("vload", inp_values + i, inp_val_offsets + i // VLEN)
                    inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("load", slots), inp_val_instr_idxs[i // VLEN])

                    inp_val_instr_idxs[i // VLEN] = max(next_instr_idxs[i // VLEN], inp_val_instr_idxs[i // VLEN])

                # check input indices / values indexed in full batch
                for j in range(i,i+VLEN):
                    self.interleave_engine_fns(body, ("debug", ("compare", inp_indices + j, (round, st + j, "idx"))), inp_val_instr_idxs[i // VLEN])
                    self.interleave_engine_fns(body, ("debug", ("compare", inp_values + j, (round, st + j, "val"))), inp_val_instr_idxs[i // VLEN])

                # if i == VLEN:
                #     inp_val_instr_idxs = self.build_apply_node_val_mem(body, i, inp_val_instr_idxs, inp_indices, inp_values, node_vals, round, st, end)
                if depth == 0:
                    inp_val_instr_idxs = self.build_apply_node_val_root(body, i, inp_val_instr_idxs, inp_values, tree_vals_vlen[0])
                elif depth < n_tree_preload_layers:
                    inp_val_instr_idxs = self.build_apply_node_val_masked(body, i, inp_val_instr_idxs, inp_values, inp_indices, node_vals, tmp1_parallel, tree_vals_vlen, forest_consts_vlen, consts_vlen, round, depth, chunk_len)
                else:
                    inp_val_instr_idxs = self.build_apply_node_val_mem(body, i, inp_val_instr_idxs, inp_indices, inp_values, node_vals, round, st, end, debug_info)

                inp_val_instr_idxs = self.build_hash_opt(body, i, inp_val_instr_idxs, inp_values, tmp1_parallel, hash_consts_vlen, round, st, end, debug_info)
                for j in range(i,i+VLEN):
                    self.interleave_engine_fns(body,("debug", ("compare", inp_values + j, (round, st + j, "hashed_val"))), inp_val_instr_idxs[i // VLEN])

                
                if round == rounds - 1:
                    # use vstore operations to write the inputs back to memory
                    slots = ("vstore", inp_val_offsets + (i // VLEN), inp_values + i)
                    inp_val_instr_idxs[i // VLEN] = self.interleave_engine_fns(body, ("store", slots), inp_val_instr_idxs[i // VLEN])

                    # no need for index update on last update
                    return

                # if at full depth, set idx to 0
                if (round + 1) % (forest_height + 1) == 0:
                    inp_val_instr_idxs = self.build_idx_wrap(body, i, inp_val_instr_idxs, inp_indices, end - st, forest_consts_vlen[0])
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                else:
                    inp_val_instr_idxs = self.build_idx_next(body, i, inp_val_instr_idxs, inp_indices, inp_values, tmp1_parallel, end - st, forest_const_m1_vlen, consts_vlen[1], consts_vlen[2])

                for j in range(i,i+VLEN):
                    self.interleave_engine_fns(body,("debug", ("compare", inp_indices + j, (round, st + j, "wrapped_idx"))), inp_val_instr_idxs[i // VLEN])

                # on last round, can potentially skip index update
            

            # go through first 3 rounds vector by vector, then process the chunk in parallel
            # schedule = [(range(0,3), "vector"), (range(3,10), "chunk"), (range(10,13), "vector"), (range(13,16), "chunk")]
            switch_point = 14
            schedule = [(range(0,switch_point), "vector"), (range(switch_point,16), "chunk")]
            for round_range, process_algo in schedule:
                if process_algo == "chunk":
                    for round in round_range:
                        for i in range(0,chunk_len,VLEN):
                            process_vector(round,i, inp_val_instr_idxs)
                elif process_algo == "vector":
                    for i in range(0,chunk_len,VLEN):
                        for round in round_range:
                            process_vector(round,i, inp_val_instr_idxs)


        print("Total scratch used: ", self.scratch_ptr, "remaining: ", SCRATCH_SIZE - self.scratch_ptr)
        
        # print(body)
        self.instrs.extend(body)
        # self.print_instructions()
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
