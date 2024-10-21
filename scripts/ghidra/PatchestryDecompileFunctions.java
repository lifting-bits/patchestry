/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import ghidra.app.script.GhidraScript;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.plugin.processors.sleigh.SleighLanguage;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressSpace;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.data.DataType;

import ghidra.program.model.lang.Language;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;

import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

import ghidra.program.model.listing.Program;

import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighSymbol;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.Varnode;

import ghidra.pcode.pcoderaw.PcodeOpRaw;

import ghidra.program.model.symbol.ExternalManager;
import ghidra.program.model.symbol.Symbol;

import com.google.gson.stream.JsonWriter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.OutputStreamWriter;
import java.io.File;

import java.nio.file.Files;
import java.nio.file.Path;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.TreeMap;

public class PatchestryDecompileFunctions extends GhidraScript {
	
	protected static final int DECLARE_LOCAL_VAR = 1000; 
	
	protected class DefinitionVarnode extends Varnode {
		private PcodeOp def;
		private HighVariable high;
		
		public DefinitionVarnode(Address address, int size) {
			super(address, size);
		}
		
		public void setDef(HighVariable high, PcodeOp def) {
			this.def = def;
			this.high = high;
		}
		
		@Override
		public PcodeOp getDef() {
			return def;
		}
		
		@Override
		public HighVariable getHigh() {
			return high;
		}
		
		@Override
		public boolean isInput() {
			return false;
		}
	};

    private class PcodeSerializer extends JsonWriter {
    	private String arch;
    	private AddressSpace constant_space;
    	private AddressSpace unique_space;
    	private FunctionManager fm;
    	private ExternalManager em;
    	private DecompInterface ifc;
    	private BasicBlockModel bbm;
    	private List<Function> functions;
    	private int original_functions_size;
    	private Set<Address> seen_functions;
    	private HighFunction current_function;
    	private PcodeBlockBasic current_block;
    	private Set<String> happens_after;
    	private Set<String> seen_indirects;
    	private Map<Address, HighSymbol> seen_addresses;
    	private Map<PcodeOp, String> labels;
    	private long next_unique;
    	private int next_seqnum;
    	private List<PcodeOp> entry_block;

        public PcodeSerializer(java.io.BufferedWriter writer,
        					   String arch_, FunctionManager fm_,
        					   ExternalManager em_, DecompInterface ifc_,
        					   BasicBlockModel bbm_,
        					   List<Function> functions_) {
            super(writer);

            Program program = fm_.getProgram();
            SleighLanguage language = (SleighLanguage) program.getLanguage();
            AddressFactory address_factory = language.getAddressFactory();

            this.arch = arch_;
            this.constant_space = address_factory.getConstantSpace();
            this.unique_space = address_factory.getUniqueSpace();
            this.fm = fm_;
            this.em = em_;
            this.ifc = ifc_;
            this.bbm = bbm_;
            this.functions = functions_;
            this.original_functions_size = functions.size();
            this.seen_functions = new TreeSet<>();
            this.current_function = null;
            this.current_block = null;
            this.happens_after = new TreeSet<>();
            this.seen_indirects = new TreeSet<>();
            this.seen_addresses = new TreeMap<>();
            this.next_unique = language.getUniqueBase();
            this.next_seqnum = 0;
            this.entry_block = new ArrayList<>();
        }
        
//        private PcodeOp representativeOf(Varnode node) throws Exception {
//        	
//        }
        
        private static String label(Address address) throws Exception {
    		return address.toString(true  /* show address space prefix */);
        }

        private static String label(SequenceNumber sn) throws Exception {
        	return label(sn.getTarget()) + Address.SEPARATOR +
        		   Integer.toString(sn.getTime()) + Address.SEPARATOR +
        		   Integer.toString(sn.getOrder());
        }
        
        private static String label(PcodeBlock block) throws Exception {
        	return label(block.getStart()) + Address.SEPARATOR +
        		   Integer.toString(block.getIndex()) + Address.SEPARATOR +
        		   PcodeBlock.typeToName(block.getType());
        }
        
        private static String label(PcodeOp op) throws Exception {
        	return label(op.getSeqnum());
        }
        
        // Try to resolve an operand `node` of an `indirect_op` to its definition.
        // This function is not resoponsible f
        private PcodeOp resolveIndirect(PcodeOp indirect_op, Varnode node) throws Exception {  	
        	if (indirect_op == null) {
        		return null;
        	}
        	
        	if (indirect_op.getOpcode() != PcodeOp.INDIRECT) {
        		return null;
        	}

        	if (node == null) {
        		return null;
        	}

        	Varnode output = indirect_op.getOutput();
        	if (output == null) {
        		return null;
        	}

        	// If this indirect actually affects a global variable then we
    		// "don't care", insofar as 
        	switch (output.getSpace() & AddressSpace.ID_TYPE_MASK) {
	        	case AddressSpace.TYPE_REGISTER:
	        	case AddressSpace.TYPE_STACK:
	        	case AddressSpace.TYPE_JOIN:
	        	case AddressSpace.TYPE_VARIABLE:
	        		break;
        		default:
        			return null;
        	}

        	PcodeOp dependent_op = null;
    		PcodeOp def_of_node = node.getDef();
        	
        	// If it's a constant, then the address of the dependent operation
        	// is the same as `indirect_op`, and `node.getOffset()` encodes
        	// the "time" of the p-code operation in terms of the sequence
        	// number. Typically we observe this with `INDIRECT`s that follow
        	// `CALL`s, where all of the `INDIRECT`s are exposing that the `CALL`
        	// likely modifies some data.
        	switch (node.getSpace() & AddressSpace.ID_TYPE_MASK) {
        		case AddressSpace.TYPE_CONSTANT:
        			if (node.getOffset() != 0 || indirect_op.getInput(0) != node) {
			        	SequenceNumber indirect_loc = indirect_op.getSeqnum();
			    		SequenceNumber dependent_loc = new SequenceNumber(
							indirect_loc.getTarget(), (int) node.getOffset());
			    		dependent_op = current_function.getPcodeOp(dependent_loc);
			    		
			    		println("=== Resolving " + node.toString());
			    		println("indirect_op: " + label(indirect_op) + "| " + indirect_op.toString());
			    		if (dependent_op != null) {
			    			println("dependent_op: " + label(indirect_op) + "| " + dependent_op.toString());
			    		}
			    		if (def_of_node != null) {
			    			println("def: " + label(def_of_node) + "| " + def_of_node.toString());
			    		}
						assert dependent_op != indirect_op;
	        		}
					break;
        		
        		case AddressSpace.TYPE_UNIQUE:
    			case AddressSpace.TYPE_REGISTER:
        		case AddressSpace.TYPE_VARIABLE:
        			dependent_op = def_of_node;
        			break;
    			
        		case AddressSpace.TYPE_STACK:
        			println("??? Resolving " + node.toString());
		    		println("indirect_op: " + label(indirect_op) + "| " + indirect_op.toString());
		    		if (def_of_node != null) {
		    			println("def: " + label(def_of_node) + "| " + def_of_node.toString());
		    		}
        			break;

        		default:
        			break;
        	}
        	
    		assert dependent_op != indirect_op;
    		return dependent_op;
        }
        
        // Collect the happens-after relationships implied by `INDIRECT` opcodes.
        //
        // TODO(pag): Can these be cyclic?
        private void collectHappensAfter(PcodeOp def) throws Exception {
        	
        	if (def == null) {
        		return;
        	}
        	
        	// There is a cycle in the `INDIRECT`s. This can happen when there's
        	// a loop.
        	if (!seen_indirects.add(label(def))) {
        		return;
        	}
        	
        	// Bottom of recursive case; we've followed indirects to something
        	// that isn't an indirect.
        	if (def.getOpcode() != PcodeOp.INDIRECT) {
        		happens_after.add(label(def));
    			return;
        	}
        	
        	collectHappensAfter(resolveIndirect(def, def.getInput(0)));
        	collectHappensAfter(resolveIndirect(def, def.getInput(1)));
        }
        
        // Resolve to the output of `b` if `a` and `b`'s output share the same
        // high variable.
        private static Varnode resolveOutputIfSameHigh(Varnode a, PcodeOp b) {
        	if (a == null || b == null) {
        		return null;
        	}
        	Varnode b_out = b.getOutput();
        	if (b_out == null) {
        		return null;
        	}

        	return a.getHigh() == b_out.getHigh() ? b_out : null;
        }
        
        // Return the r-value of a varnode.
        private Varnode rValueOf(Varnode node) throws Exception {
        	collectHappensAfter(node.getDef());
        	
        	HighVariable high = node.getHigh();
        	if (high == null) {
        		return node;
        	}
        	
        	Varnode rep = high.getRepresentative();
        	return rep == null ? node : rep;
        }

        private void serializeInput(Varnode node) throws Exception {
        	assert node.isInput();
        	assert !node.isFree();

    		PcodeOp def = node.getDef();

            beginObject();
            name("size").value(node.getSize());

        	if (node.isUnique()) {
        		assert def != null;
        		name("kind").value("operation");
        		name("operation").value(label(def));
        	}
            endObject();
        }
        
        private void serialize(Varnode node) throws Exception {
            if (node == null) {
            	assert false;
                nullValue();
                return;
            }
            
            // Make sure INDIRECTs don't leak back into our output. We won't
            // have the ability to reference them.
            PcodeOp def = node.getDef();
            if (def != null) {
            	assert def.getOpcode() != PcodeOp.INDIRECT;
            }

            beginObject();

            if (node.isConstant()) {
                name("type").value("const");
            } else if (node.isUnique()) {
                name("type").value("unique");
            } else if (node.isRegister()) {
                name("type").value("register");
            } else if (node.isAddress()) {
                name("type").value("ram");
            } else if (node.getAddress().isStackAddress()) {
                name("type").value("stack");
            } else {
                throw new Exception("Unknown Varnode kind: " + node.toString());
            }

            name("offset").value(node.getOffset());
            name("size").value(node.getSize());

            endObject();
        }
        
//        private AddressSpace getAddressSpace(long offset) {
//        	switch (((int) offset) & AddressSpace.ID_TYPE_MASK) {
//				case AddressSpace.TYPE_RAM:
//					return AddressSpace.
//        	}
//        }
        
        // The address of a `LOAD` or `STORE` is spread across two operands:
        // the first being a constant representing the address space, and the
        // second being the actual address.
        private void serializeLoadStoreAddress(PcodeOp op) throws Exception {
        	Varnode aspace = op.getInput(0);
        	assert aspace.isConstant();

        	Varnode address = rValueOf(op.getInput(1));

        	beginObject();
            name("size").value(op.getInput(1).getSize());
            println(op.toString());
            //        	if (address.isConstant()) {
//        		AddressSpace address_space = null;
//        		
//        		Address address = new Address();
//        	}

        	endObject();
        }
        
        private void recordAddress(Address address) {
        	seen_addresses.put(address, null);
        }
        
        private void recordAddress(Address address, HighVariable var) {
        	if (!address.isMemoryAddress() && !address.isExternalAddress()) {
        		println("$$$ high variable " + var.toString() + " at " + address.toString());
        		return;
        	}
        	
        	HighSymbol symbol = var.getSymbol();
        	if (symbol == null) {
        		println("!!! high variable no symbol " + var.toString() +" at " + address.toString());
        	}
//        	
//        	
//	        	seen_addresses.put(symbol.getPCAddress(), symbol);
//	        	int offset = var.getOffset();
//	        	if (-1 == offset) {
//	        		
//	        	} else {
//	        		
//	        	}
//        	}
        }
        
        private Address nextUniqueAddress() throws Exception {
        	Address address = unique_space.getAddress(next_unique);
        	next_unique += unique_space.getAddressableUnitSize();
        	return address;
        }
        
        // Creates a pseudo p-code op using a `CALLOTHER` that logically
        // represents the definition of a local variable.
        private PcodeOp createLocalVarDecl(HighVariable var) throws Exception {
        	HighSymbol high_symbol = var.getSymbol();
        	Address address = nextUniqueAddress();
        	DefinitionVarnode def = new DefinitionVarnode(address, var.getSize());
        	Varnode[] ins = new Varnode[2];
        	SequenceNumber loc = new SequenceNumber(address, next_seqnum++);
        	PcodeOp op = new PcodeOp(loc, PcodeOp.CALLOTHER, 1, def);
        	op.insertInput(new Varnode(constant_space.getAddress(DECLARE_LOCAL_VAR), 4), 0);
        	def.setDef(var, op);

        	Varnode[] instances = var.getInstances();
        	Varnode[] new_instances = new Varnode[instances.length + 1];
        	System.arraycopy(instances, 0, new_instances, 1, instances.length);
        	new_instances[0] = def;

        	var.attachInstances(new_instances, def);
        	
        	entry_block.add(op);
        	
        	return op;
        }
        
        // Get or create a local variable pseudo definition op for the high
        // variable `var`.
        private PcodeOp getOrCreateLocalVariable(HighVariable var) throws Exception {
        	Varnode representative = var.getRepresentative();
        	if (representative == null) {
        		return createLocalVarDecl(var);
        	}
        	
        	PcodeOp def = representative.getDef();
        	if (def == null) {
        		return createLocalVarDecl(var);
        	}
        	
        	if (def.getOpcode() != PcodeOp.CALLOTHER) {
        		return createLocalVarDecl(var);
        	}
        	
        	if (def.getInput(0).getOffset() != DECLARE_LOCAL_VAR) {
        		return createLocalVarDecl(var);
        	}
        	
        	return def;
        }
        
        // Handles serializing the output, if any, of `op`. 
        private void serializeOutput(PcodeOp op) throws Exception {
        	Varnode output = op.getOutput();
        	if (output == null) {
        		return;
        	}

        	name("output").beginObject();

        	HighVariable var = output.getHigh();
        	if (var != null) {
        		name("kind").value("local_variable");
        		name("variable").value(label(getOrCreateLocalVariable(var)));
        	} else {
        		
        	}
        	endObject();
//        	
//    		Varnode representative = var.getRepresentative();
//    		
//    		
//    		if (representative != null) {
//    			PcodeOp representative_def = representative.getDef();
//    			if (representative_def != null) {
//    				if (representative_def == op) {
//    					name("kind").value("self");
//    				} else {
//        				name("kind").value("operation");
//        				name("operation").value(label(representative_def));
//    				}
//    			} else if (representative.isAddress()) {
//    				recordAddress(representative.getAddress(), var);
//    				name("kind").value("memory");
//    				name("address").value(label(output.getAddress()));
//    				//name("symbol").value(label(representative.get));
//
//    			} else {
//    				println("!!! representative: " + var.toString() + " is node " + representative.toString());
//    			}
//    		} else {
//    			println("!!! representative: " + var.toString());
//    		}
//        	
//        	endObject();
        }

        // Serialize a direct call. This enqueues the targeted for type lifting
        // `Function` if it can be resolved.
        private void serializeDirectCallOp(PcodeOp op) throws Exception {
        	Address caller_address = current_function.getFunction().getEntryPoint();
        	Varnode target_address_node = op.getInput(0);
    		if (!target_address_node.isAddress()) {
    			throw new Exception("Unexpected non-address input to CALL");
    		}

    		Address target_address = caller_address.getNewAddress(target_address_node.getOffset());
    		String target_label = label(target_address);
    		Function callee = fm.getFunctionAt(target_address);
    		
    		// `target_address` may be a pointer to an external. Figure out
    		// what we're calling.
    		if (callee == null) {
    			callee = fm.getReferencedFunction(target_address);
    			if (callee != null) {
    				target_address = callee.getEntryPoint();
    				println("Call through " + target_label +
    						" targets " + callee.getName() +
    						" at " + label(target_address));
    				target_label = label(target_address);
    			}
    		}
    		
    		name("target_address").value(target_label);

    		if (callee != null) {
    			functions.add(callee);
    		} else {
    			println("Could not find function at address " + target_label +
    					" called by " + caller_address.toString());
    		}
        }
        
        // Serialize an unconditional branch. This records the targeted block.
        private void serializeBranchOp(PcodeOp op) throws Exception {
        	assert current_block.getOutSize() == 1;
    		name("target_block").value(label(current_block.getOut(0)));
        }
        
        // Serialize a conditional branch. This records the targeted blocks.
        //
        // TODO(pag): How does p-code handle delay slots? Are they separate
        //		      blocks?
        //
        // XREF(pag): https://github.com/NationalSecurityAgency/ghidra/issues/2736
        //			  describes how the `op` meaning of the branch, i.e. whether
        //			  it branches on zero or not zero, can change over the course
        //			  of simplification, and so the inputs representing the
        //			  branch targets may not actually represent the `true` or
        //			  `false` outputs in the traditional sense.
        private void serializeCondBranchOp(PcodeOp op) throws Exception {
        	name("taken_block").value(label(current_block.getTrueOut()));
        	name("not_taken_block").value(label(current_block.getFalseOut()));
        	name("condition");
        	serializeInput(rValueOf(op.getInput(1)));
        }
        
        // Serialize a generic multi-input, single-output p-code operation.
        private void serializeGenericOp(PcodeOp op) throws Exception {
            name("inputs").beginArray();
            for (Varnode input : op.getInputs()) {
            	serializeInput(rValueOf(input));
            }
            endArray();
        }
        
        // Serializes a pseudo-op `DECLARE_LOCAL_VAR`, which is actually encoded
        // as a `CALLOTHER`.
        private void serializeDeclareLocalVar(PcodeOp op) throws Exception {
        	name("name").value(op.getOutput().getHigh().getName());
        }
        
        // Serialize a `CALLOTHER`. The first input operand is a constant
        // representing the user-defined opcode number. In our case, we have
        // our own user-defined opcodes for making things better mirror the
        // structure/needs of MLIR.
        private void serializeCallOtherOp(PcodeOp op) throws Exception {
        	switch ((int) op.getInput(0).getOffset()) {
				case DECLARE_LOCAL_VAR:
					serializeDeclareLocalVar(op);
					break;
				default:
					serializeGenericOp(op);
					break;
        	}
        }
        
        // Get the mnemonic for a p-code operation. We have some custom
        // operations encoded as `CALLOTHER`s, so we get their names manually
        // here.
        //
        // TODO(pag): There is probably a way to register the name of a
        //			  `CALLOTHER` via `Language.getSymbolTable()` using a
        //			  `UseropSymbol`. It's not clear if there's really value in
        //			  doing this, though.
        private static String mnemonic(PcodeOp op) {
        	if (op.getOpcode() == PcodeOp.CALLOTHER) {
        		switch ((int) op.getInput(0).getOffset()) {
        			case DECLARE_LOCAL_VAR:
        				return "DECLARE_LOCAL_VAR";
    				default:
    					break;
        		}
        	}
        	return op.getMnemonic();
        }

        private void serialize(PcodeOp op) throws Exception {
    	
            beginObject();
            name("mnemonic").value(mnemonic(op));
        	serializeOutput(op);
            switch (op.getOpcode()) {
            	case PcodeOp.CALL:
            		serializeDirectCallOp(op);
            		break;
            	case PcodeOp.CALLOTHER:
            		serializeCallOtherOp(op);
            		break;
            	case PcodeOp.CBRANCH:
            		serializeCondBranchOp(op);
                	break;
            	case PcodeOp.BRANCH:
            		serializeBranchOp(op);
                	break;
            	case PcodeOp.LOAD:
            	case PcodeOp.COPY:
            	case PcodeOp.CAST:
            	default:
            		serializeGenericOp(op);
            		break;
            }

            // Serialize the list of operations that indirectly affected the
            // operands of this operation. This operation is said to "happen
            // after" these other indirect operations.
            if (!happens_after.isEmpty()) {
            	name("happens_after").beginArray();
            	for (String pred_label : happens_after) {
            		value(pred_label);
            	}
            	endArray();
            	happens_after.clear();
            }
        	seen_indirects.clear();

            endObject();
        }
        
        // Serialize a high p-code basic block. This iterates over the p-code
        // operations within the block and serializes them individually.
        private void serialize(PcodeBlockBasic block) throws Exception {
        	PcodeBlock parent_block = block.getParent();
        	if (parent_block != null) {
        		name("parent_block").value(label(parent_block));
        	}
        	
            PcodeOp op = null;
            Iterator<PcodeOp> op_iterator = block.getIterator();
            name("operations").beginObject();
            while (op_iterator.hasNext()) {
            	op = op_iterator.next();
            	
            	// NOTE(pag): INDIRECTs seem like a good way of modelling may-
            	//		      alias relations, as well as embedding control
            	//			  dependencies into the dataflow graph, e.g. to
            	//			  ensure code motion cannot happen from after a CALL
            	//			  to before a CALL, especially for stuff operating
            	//			  on stack slots. The idea at the time of this
            	//			  comment is that we will assume that eventual
            	//			  codegen also should not do any reordering, though
            	//			  enforcing that is also tricky.
            	if (op.getOpcode() != PcodeOp.INDIRECT) {
            		name(label(op));
            		current_block = block;
            		serialize(op);
            		current_block = null;
            	}
            }
            endObject();
            
            // List out the operations in their order.
            op_iterator = block.getIterator();
            name("ordered_operations").beginArray();
            while (op_iterator.hasNext()) {
            	value(label(op_iterator.next()));
            }
            endArray();
        }
        
        // Emit a pseudo entry block to represent
        private void serializeEntryBlock(PcodeBlockBasic first_block) throws Exception {
        	name("entry").beginObject();
        	name("operations").beginObject();
        	for (PcodeOp pseudo_op : entry_block) {
        		name(label(pseudo_op));
        		serialize(pseudo_op);
        	}

        	// If there is a proper entry block, then invent a branch to it.
        	if (first_block != null) {
	        	name("entry.exit").beginObject();
	        	name("mnemonic").value("BRANCH");
	        	name("target_block").value(label(first_block));
	        	endObject();  // End of BRANCH to `first_block`.
        	}

        	endObject();  // End of operations.

        	name("ordered_operations").beginArray();
        	for (PcodeOp pseudo_op : entry_block) {
        		value(label(pseudo_op));
        	}
        	value("entry.exit");
        	endArray();  // End of `ordered_operations`.
        	endObject();  // End of `entry` block.
        }

        // Serialize `function`. If we have `high_function` (the decompilation
        // of function) then we will serialize its type information. Otherwise,
        // we will serialize the type information of `function`. If
        // `visit_pcode` is true, then this is a function for which we want to
        // fully lift, i.e. visit all the high p-code.
        private void serialize(HighFunction high_function, Function function, boolean visit_pcode) throws Exception {
            
            name("name").value(function.getName());
            
            // If we have a high P-Code function, then serialize the blocks.
            if (high_function == null || !visit_pcode) {
            	return;
            }
    		PcodeBlockBasic first_block = null;
            current_function = high_function;
            name("basic_blocks").beginObject();
            for (PcodeBlockBasic block : high_function.getBasicBlocks()) {
            	if (first_block == null) {
            		first_block = block;
            	}
            	name(label(block)).beginObject();
                serialize(block);
                endObject();
            }
            
            // If we created a fake entry block to represent variable
            // declarations then emit that here.
            if (!entry_block.isEmpty()) {
            	serializeEntryBlock(first_block);
            }

            endObject();
            current_function = null;
            
            if (!entry_block.isEmpty()) {
        		name("entry_block").value("entry");
            	entry_block.clear();

            } else if (first_block != null) {
            	name("entry_block").value(label(first_block));
            }
        }

        // Serialize the input function list to JSON. This function will also
        // serialize type information related to referenced functions and
        // variables.
        public void serialize() throws Exception {

            beginObject();
            name("arch").value(getArch());
            name("format").value(currentProgram.getExecutableFormat());
            name("functions").beginObject();
            
            for (int i = 0; i < functions.size(); ++i) {
            	Function function = functions.get(i);
            	Address function_address = function.getEntryPoint();
            	if (!seen_functions.add(function_address)) {
            		continue;
            	}

        		DecompileResults res = ifc.decompileFunction(function, 30, null);
        		HighFunction high_function = res.getHighFunction();

        		name(label(function_address)).beginObject();
        		serialize(high_function, function, i < original_functions_size);
        		endObject();
            }
            endObject().endObject();
        }
    }

    private String getArch() throws Exception {
        if (currentProgram.getLanguage() == null ||
		    currentProgram.getLanguage().getProcessor() == null) {
            return "unknown";
        }
        return currentProgram.getLanguage().getProcessor().toString();
    }
    
    private DecompInterface getDecompilerInterface() throws Exception {
        if (currentProgram == null) {
            throw new Exception("Unable to initialize decompiler: invalid current program.");
        }
        DecompInterface decompiler = new DecompInterface();
        decompiler.setOptions(new DecompileOptions());
        if (!decompiler.openProgram(currentProgram)) {
            throw new Exception("Unable to initialize decompiler: " + decompiler.getLastMessage());
        }
        return decompiler;
    }
    
    private void serializeToFile(Path file, List<Function> functions) throws Exception {
        if (file == null || functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Invalid file path or empty function list");
        }
 
        final var serializer = new PcodeSerializer(
    		Files.newBufferedWriter(file), getArch(),
    		currentProgram.getFunctionManager(), currentProgram.getExternalManager(),
    		getDecompilerInterface(), new BasicBlockModel(currentProgram), functions);
        serializer.serialize();
        serializer.close();
    }

    private List<Function> getAllFunctions() {
        if (currentProgram == null || currentProgram.getFunctionManager() == null) {
            return Collections.emptyList();
        }
        FunctionIterator functionIter = currentProgram.getFunctionManager().getFunctions(true);
        List<Function> functions = new ArrayList<>();
        while (functionIter.hasNext() && !monitor.isCancelled()) {
            functions.add(functionIter.next());
        }
        return functions;
    }

    private void decompileSingleFunction() throws Exception {
        if (getScriptArgs().length < 3) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <function_name> <output_file> as argument");
        }
        serializeToFile(Path.of(getScriptArgs()[2]), getGlobalFunctions(getScriptArgs()[1]));
    }

    private void decompileAllFunctions() throws Exception {
        if (getScriptArgs().length < 2) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <output_file> as argument");
        }
        serializeToFile(Path.of(getScriptArgs()[1]), getAllFunctions());
    }

    private void runHeadless() throws Exception {
        if (getScriptArgs().length < 1) {
            throw new IllegalArgumentException("mode is not specified for headless execution");
        }

        // Execution mode
        String mode = getScriptArgs()[0];
        println("Running in mode: " + mode);
        switch (mode.toLowerCase()) {
            case "single":
                decompileSingleFunction();
                break;
            case "all":
                decompileAllFunctions();
                break;
            default:
                throw new IllegalArgumentException("Invalid mode: " + mode);
        }
    }

    private void decompileSingleFunctionInGUI() throws Exception {
    	List<Function> functions = null;
    	if (currentProgram != null) {
	    	FunctionManager manager = currentProgram.getFunctionManager();
	    	if (manager != null) {
	        	Function function = manager.getFunctionContaining(currentAddress);
	        	if (function != null) {
	        		functions = new ArrayList<>();
	        		functions.add(function);
	        	}
	    	}
    	}
    	
    	if (functions == null) {
    		String functionNameArg = askString("functionNameArg", "Function name to decompile: ");
            functions = getGlobalFunctions(functionNameArg);
    	}
        
        File outputDirectory = askDirectory("outputFilePath", "Select output directory");
        File outputFilePath = new File(outputDirectory, "patchestry.json");
        serializeToFile(outputFilePath.toPath(), functions);
    }

    private void decompileAllFunctionsInGUI() throws Exception {
        File outputDirectory = askDirectory("outputFilePath", "Select output directory");
        File outputFilePath = new File(outputDirectory, "patchestry.json");
        serializeToFile(outputFilePath.toPath(), getAllFunctions());
    }

    // GUI mode execution
    private void runGUI() throws Exception {
        String mode = askString("mode", "Please enter mode:");
        println("Running in mode: " + mode);
        switch (mode.toLowerCase()) {
            case "single":
                decompileSingleFunctionInGUI();
                break;
            case "all":
                decompileAllFunctionsInGUI();
                break;
            default:
                throw new IllegalArgumentException("Invalid mode: " + mode);
        }
    }

    // Script entry point
    @Override
    public void run() throws Exception {
        try {
            if (isRunningHeadless()) {
                runHeadless();
            } else {
                runGUI();
            }
        } catch (Exception e) {
            println("Error: " + e.getMessage());
            e.printStackTrace(new PrintWriter(new OutputStreamWriter(System.err)));
            throw e;
        }
    }
}
