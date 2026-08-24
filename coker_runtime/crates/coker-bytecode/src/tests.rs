use super::*;
use rkyv::util::AlignedVec;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.with(Cell::get) {
            ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK_ALLOCATIONS.with(Cell::get) {
            ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.with(Cell::get) {
            ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        System.alloc_zeroed(layout)
    }
}

struct AllocationTracker;

impl AllocationTracker {
    fn start() -> Self {
        TRACK_ALLOCATIONS.with(|flag| flag.set(true));
        ALLOCATION_COUNT.with(|count| count.set(0));
        Self
    }

    fn count() -> usize {
        ALLOCATION_COUNT.with(Cell::get)
    }
}

impl Drop for AllocationTracker {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.with(|flag| flag.set(false));
    }
}

fn single_function_module() -> BytecodeModule {
    BytecodeModule::new(vec![Program::new(0, 0, vec![], vec![], vec![])])
}
fn complex_module() -> BytecodeModule {
    let callee_program = Program::new(
        3,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 2,
            length: 1,
        }],
        vec![Layer::Generic(GenericLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
            out_length: 2,
            scratch_offset: 0,
            scratch_length: 0,
            ops: vec![
                RowOp {
                    first: 0,
                    second: 0,
                    third: 0,
                    op: ScalarOp::Identity,
                },
                RowOp {
                    first: 0,
                    second: 0,
                    third: 0,
                    op: ScalarOp::Sin,
                },
            ],
        })],
    );
    let entry_program = Program::new(
        1,
        4,
        vec![],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![Layer::Evaluate(EvaluateLayer {
            scratch_offset: 1,
            callee_function_id: 1,
            input_bindings: vec![EvaluateInputBinding::ConstantSlice {
                length: 1,
                values: vec![2.0],
            }],
            output_bindings: vec![EvaluateOutputBinding {
                destination_offset: 0,
                length: 1,
            }],
        })],
    );
    BytecodeModule::new(vec![entry_program, callee_program])
}

#[test]
fn encode_decode_round_trip_preserves_module() {
    let tensor_data: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 0.0], [0.0, 4.0, 0.0, 5.0]];
    let tensor = SparseTensor::try_from_row_major_array(&tensor_data).unwrap();
    let entry_program = Program::new(
        6,
        9,
        vec![InputSpec {
            workspace_offset: 0,
            length: 3,
        }],
        vec![OutputSpec {
            workspace_offset: 4,
            length: 2,
        }],
        vec![
            Layer::Bilinear(BilinearLayer {
                in_offset: 0,
                out_offset: 4,
                in_length: 3,
                out_length: 2,
                scratch_offset: 0,
                scratch_length: 0,
                quadratic: tensor,
            }),
            Layer::Evaluate(EvaluateLayer {
                scratch_offset: 6,
                callee_function_id: 1,
                input_bindings: vec![EvaluateInputBinding::WorkspaceSlice {
                    offset: 4,
                    length: 2,
                }],
                output_bindings: vec![EvaluateOutputBinding {
                    destination_offset: 4,
                    length: 2,
                }],
            }),
        ],
    );
    let callee_program = Program::new(
        2,
        2,
        vec![InputSpec {
            workspace_offset: 0,
            length: 2,
        }],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 2,
        }],
        vec![],
    );
    let module = BytecodeModule::new(vec![entry_program, callee_program]);

    let encoded_module = encode_module(&module).unwrap();
    let decoded_module = decode_module(&encoded_module).unwrap();
    assert_eq!(decoded_module, module);
}
#[test]
fn encode_archives_pad_payloads_to_declared_alignment() {
    let module = BytecodeModule::new(vec![Program::new(0, 0, vec![], vec![], vec![])]);
    let module_encoded = encode_module(&module).unwrap();
    let module_alignment = u16::from_le_bytes(
        module_encoded[MAGIC.len() + 2..MAGIC.len() + 4]
            .try_into()
            .expect("module header stores alignment bytes"),
    ) as usize;
    let module_payload_offset = payload_start_offset(module_alignment);
    assert_eq!(
        &module_encoded[MAGIC.len() + 2..MAGIC.len() + 4],
        &(ARCHIVED_MODULE_ALIGNMENT as u16).to_le_bytes()
    );
    assert_eq!(module_payload_offset % module_alignment, 0);
    assert!(module_encoded[HEADER_SIZE..module_payload_offset]
        .iter()
        .all(|byte| *byte == 0));
    assert_eq!(decode_module(&module_encoded).unwrap(), module);
}

#[test]
fn archived_module_accepts_aligned_bytes_without_copy() {
    let module = single_function_module();
    let encoded = encode_module(&module).unwrap();
    let mut aligned = AlignedVec::<16>::with_capacity(encoded.len());
    aligned.extend_from_slice(&encoded);

    let _tracker = AllocationTracker::start();
    let archived = archived_module(aligned.as_slice()).unwrap();
    assert_eq!(AllocationTracker::count(), 0);

    let payload_start = aligned.as_slice()[HEADER_SIZE..].as_ptr() as usize;
    let payload_len = aligned.as_slice().len() - HEADER_SIZE;
    let archived_ptr = archived as *const ArchivedBytecodeModule as usize;
    assert!(archived_ptr >= payload_start);
    assert!(archived_ptr - payload_start < payload_len);
    assert!(archived.entry_program().is_some());
    assert!(archived.program(0).is_some());
}

#[test]
fn archived_program_views_borrow_from_mapped_bytes() {
    let entry_program = Program::new(0, 0, vec![], vec![], vec![]);
    let generic_program = Program::new(
        12,
        24,
        vec![InputSpec {
            workspace_offset: 1,
            length: 2,
        }],
        vec![OutputSpec {
            workspace_offset: 4,
            length: 1,
        }],
        vec![Layer::Generic(GenericLayer {
            in_offset: 1,
            out_offset: 4,
            in_length: 2,
            out_length: 1,
            scratch_offset: 6,
            scratch_length: 0,
            ops: vec![RowOp {
                first: 0,
                second: 1,
                third: u16::MAX,
                op: ScalarOp::Add,
            }],
        })],
    );
    let module = BytecodeModule::new(vec![entry_program, generic_program]);
    let encoded = encode_module(&module).unwrap();
    let mut aligned = AlignedVec::<{ ARCHIVED_MODULE_ALIGNMENT }>::with_capacity(encoded.len());
    aligned.extend_from_slice(&encoded);

    let _tracker = AllocationTracker::start();
    let archived = archived_module(aligned.as_slice()).unwrap();
    let entry = archived
        .entry_program()
        .expect("entry program should exist");
    let program = archived.program(1).expect("function id 1 should exist");

    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(archived.programs().count(), 2);
    let first_program = archived.programs().next().unwrap().1;
    assert!(core::ptr::eq(first_program, entry));
    assert_eq!(entry.workspace_size(), 0);
    assert_eq!(entry.required_workspace_size(), 0);
    assert_eq!(entry.intermediate_layers().len(), 0);

    assert_eq!(program.workspace_size(), 12);
    assert_eq!(program.required_workspace_size(), 24);
    assert_eq!(program.input_specs().len(), 1);
    assert_eq!(program.output_specs().len(), 1);
    assert_eq!(program.intermediate_layers().len(), 1);

    let payload_start = aligned.as_slice()[HEADER_SIZE..].as_ptr() as usize;
    let payload_len = aligned.as_slice().len() - HEADER_SIZE;
    let program_ptr = program as *const ArchivedProgram as usize;
    assert!(program_ptr >= payload_start);
    assert!(program_ptr - payload_start < payload_len);

    let input_specs_ptr = program.input_specs().as_ptr() as usize;
    let layers_ptr = program.intermediate_layers().as_ptr() as usize;
    assert!(input_specs_ptr >= payload_start);
    assert!(input_specs_ptr - payload_start < payload_len);
    assert!(layers_ptr >= payload_start);
    assert!(layers_ptr - payload_start < payload_len);

    match &program.intermediate_layers()[0] {
        ArchivedLayer::Generic(generic_layer) => {
            assert_eq!(generic_layer.in_offset.to_native(), 1);
            assert_eq!(generic_layer.out_offset.to_native(), 4);
            assert_eq!(generic_layer.in_length.to_native(), 2);
            assert_eq!(generic_layer.out_length.to_native(), 1);
            assert_eq!(generic_layer.ops.len(), 1);
            let row_operation = &generic_layer.ops[0];
            assert_eq!(row_operation.first.to_native(), 0);
            assert_eq!(row_operation.second.to_native(), 1);
            assert_eq!(row_operation.third.to_native(), u16::MAX);
            assert!(matches!(row_operation.op, ArchivedScalarOp::Add));
        }
        _ => panic!("expected generic layer"),
    }

    assert_eq!(AllocationTracker::count(), 0);
}

#[test]
fn archived_module_rejects_short_header() {
    let encoded = encode_module(&single_function_module()).unwrap();
    let error = match archived_module(&encoded[..HEADER_SIZE - 1]) {
        Ok(_) => panic!("expected short header to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::TooShort)
    ));
}

#[test]
fn archived_module_rejects_magic_mismatch() {
    let mut encoded = encode_module(&single_function_module()).unwrap();
    encoded[0] ^= 0xFF;
    let error = match archived_module(&encoded) {
        Ok(_) => panic!("expected magic mismatch to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::MagicMismatch)
    ));
}

#[test]
fn archived_module_rejects_version_mismatch() {
    let mut encoded = encode_module(&single_function_module()).unwrap();
    encoded[MAGIC.len()..MAGIC.len() + 2].copy_from_slice(&(VERSION + 1).to_le_bytes());
    let error = match archived_module(&encoded) {
        Ok(_) => panic!("expected version mismatch to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::UnsupportedVersion {
            found
        }) if found == VERSION + 1
    ));
}

#[test]
fn archived_module_rejects_non_power_of_two_alignment() {
    let mut encoded = encode_module(&single_function_module()).unwrap();
    encoded[MAGIC.len() + 2..MAGIC.len() + 4].copy_from_slice(&6u16.to_le_bytes());
    let error = match archived_module(&encoded) {
        Ok(_) => panic!("expected invalid alignment to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::InvalidPayloadAlignment { actual: 6 })
    ));
}

#[test]
fn archived_module_rejects_alignment_too_small() {
    let mut encoded = encode_module(&single_function_module()).unwrap();
    encoded[MAGIC.len() + 2..MAGIC.len() + 4].copy_from_slice(&4u16.to_le_bytes());
    let error = match archived_module(&encoded) {
        Ok(_) => panic!("expected alignment check to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::PayloadAlignmentTooSmall {
            minimum: ARCHIVED_MODULE_ALIGNMENT,
            actual: 4
        })
    ));
}

#[test]
fn archived_module_rejects_truncated_payload_offset() {
    let mut encoded = encode_module(&single_function_module()).unwrap();
    encoded[MAGIC.len() + 2..MAGIC.len() + 4].copy_from_slice(&32u16.to_le_bytes());
    encoded.truncate(24);
    let error = match archived_module(&encoded) {
        Ok(_) => panic!("expected truncated payload to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::PayloadTooShort {
            expected: 32,
            actual: 24
        })
    ));
}

#[test]
fn archived_module_rejects_misaligned_payload() {
    let module = single_function_module();
    let encoded = encode_module(&module).unwrap();
    let mut misaligned = vec![0u8];
    misaligned.extend_from_slice(&encoded);
    let error = match archived_module(&misaligned[1..]) {
        Ok(_) => panic!("expected misaligned archive access to fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BytecodeError::Header(BytecodeHeaderError::PayloadMisaligned {
            expected: ARCHIVED_MODULE_ALIGNMENT,
            actual: _
        })
    ));
}
#[test]
fn archived_module_rejects_malformed_payload() {
    let mut malformed = encode_module(&complex_module()).unwrap();
    malformed[HEADER_SIZE..HEADER_SIZE + 32].fill(0xFF);
    let error = match archived_module(&malformed) {
        Ok(_) => panic!("expected malformed bytecode payload to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, BytecodeError::Decode(_)));
}
