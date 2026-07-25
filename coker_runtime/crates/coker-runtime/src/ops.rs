use coker_bytecode::{RowOp, ScalarOp};

use crate::UNUSED_OPERAND;

pub(crate) fn evaluate_generic_value(row_operation: &RowOp, input_slice: &[f32]) -> f32 {
    let first = operand_value(row_operation.first, input_slice);
    let second = operand_value(row_operation.second, input_slice);
    let third = operand_value(row_operation.third, input_slice);

    match row_operation.op {
        ScalarOp::Identity => required_operand(first),
        ScalarOp::Sin => libm::sinf(required_operand(first)),
        ScalarOp::Cos => libm::cosf(required_operand(first)),
        ScalarOp::Tan => libm::tanf(required_operand(first)),
        ScalarOp::Exp => libm::expf(required_operand(first)),
        ScalarOp::Sqrt => libm::sqrtf(required_operand(first)),
        ScalarOp::Log => libm::logf(required_operand(first)),
        ScalarOp::Neg => -required_operand(first),
        ScalarOp::Abs => libm::fabsf(required_operand(first)),
        ScalarOp::Add => required_operand(first) + required_operand(second),
        ScalarOp::Sub => required_operand(first) - required_operand(second),
        ScalarOp::Mul => required_operand(first) * required_operand(second),
        ScalarOp::Div => divide(required_operand(first), required_operand(second)),
        ScalarOp::Pow | ScalarOp::IntPow => {
            libm::powf(required_operand(first), required_operand(second))
        }
        ScalarOp::Atan2 => libm::atan2f(required_operand(first), required_operand(second)),
        ScalarOp::Equal => (required_operand(first) == required_operand(second)) as u8 as f32,
        ScalarOp::LessThan => (required_operand(first) < required_operand(second)) as u8 as f32,
        ScalarOp::LessEqual => (required_operand(first) <= required_operand(second)) as u8 as f32,
        ScalarOp::Case => {
            if required_operand(first) != 0.0 {
                required_operand(second)
            } else {
                required_operand(third)
            }
        }
    }
}

pub(crate) fn evaluate_generic_push_forward(
    row_operation: &RowOp,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
) -> (f32, f32) {
    let first = operand_value(row_operation.first, input_slice);
    let second = operand_value(row_operation.second, input_slice);
    let third = operand_value(row_operation.third, input_slice);
    let first_tangent = operand_tangent(row_operation.first, tangent_input_slice);
    let second_tangent = operand_tangent(row_operation.second, tangent_input_slice);
    let third_tangent = operand_tangent(row_operation.third, tangent_input_slice);

    match row_operation.op {
        ScalarOp::Identity => (required_operand(first), required_operand(first_tangent)),
        ScalarOp::Sin => {
            let value = required_operand(first);
            let tangent = required_operand(first_tangent);
            (libm::sinf(value), libm::cosf(value) * tangent)
        }
        ScalarOp::Cos => {
            let value = required_operand(first);
            let tangent = required_operand(first_tangent);
            (libm::cosf(value), -libm::sinf(value) * tangent)
        }
        ScalarOp::Tan => {
            let value = required_operand(first);
            let tangent = required_operand(first_tangent);
            let cos_value = libm::cosf(value);
            (libm::tanf(value), tangent / (cos_value * cos_value))
        }
        ScalarOp::Exp => {
            let value = libm::expf(required_operand(first));
            (value, value * required_operand(first_tangent))
        }
        ScalarOp::Sqrt => {
            let value = libm::sqrtf(required_operand(first));
            (value, required_operand(first_tangent) / (2.0 * value))
        }
        ScalarOp::Log => {
            let value = required_operand(first);
            (libm::logf(value), required_operand(first_tangent) / value)
        }
        ScalarOp::Neg => (-required_operand(first), -required_operand(first_tangent)),
        ScalarOp::Abs => {
            let value = required_operand(first);
            let sign = if value < 0.0 { -1.0 } else { 1.0 };
            (libm::fabsf(value), sign * required_operand(first_tangent))
        }
        ScalarOp::Add => (
            required_operand(first) + required_operand(second),
            required_operand(first_tangent) + required_operand(second_tangent),
        ),
        ScalarOp::Sub => (
            required_operand(first) - required_operand(second),
            required_operand(first_tangent) - required_operand(second_tangent),
        ),
        ScalarOp::Mul => {
            let first_value = required_operand(first);
            let second_value = required_operand(second);
            (
                first_value * second_value,
                second_value * required_operand(first_tangent)
                    + first_value * required_operand(second_tangent),
            )
        }
        ScalarOp::Div => {
            let numerator = required_operand(first);
            let denominator = required_operand(second);
            if denominator == 0.0 {
                return (f32::NAN, f32::NAN);
            }
            (
                divide(numerator, denominator),
                divide(
                    required_operand(first_tangent) * denominator
                        - numerator * required_operand(second_tangent),
                    denominator * denominator,
                ),
            )
        }
        ScalarOp::Pow | ScalarOp::IntPow => {
            let base = required_operand(first);
            let exponent = required_operand(second);
            let value = libm::powf(base, exponent);
            if base == 0.0 {
                return (value, 0.0);
            }
            (
                value,
                value
                    * (required_operand(second_tangent) * libm::logf(base)
                        + exponent * required_operand(first_tangent) / base),
            )
        }
        ScalarOp::Atan2 => {
            let first_value = required_operand(first);
            let second_value = required_operand(second);
            let denominator = first_value * first_value + second_value * second_value;
            (
                libm::atan2f(first_value, second_value),
                (second_value * required_operand(first_tangent)
                    - first_value * required_operand(second_tangent))
                    / denominator,
            )
        }
        ScalarOp::Equal => (
            (required_operand(first) == required_operand(second)) as u8 as f32,
            0.0,
        ),
        ScalarOp::LessThan => (
            (required_operand(first) < required_operand(second)) as u8 as f32,
            0.0,
        ),
        ScalarOp::LessEqual => (
            (required_operand(first) <= required_operand(second)) as u8 as f32,
            0.0,
        ),
        ScalarOp::Case => {
            if required_operand(first) != 0.0 {
                (required_operand(second), required_operand(second_tangent))
            } else {
                (required_operand(third), required_operand(third_tangent))
            }
        }
    }
}

pub(crate) fn homogeneous_value(input_slice: &[f32], operand_index: u16) -> f32 {
    if operand_index == 0 {
        1.0
    } else {
        input_slice
            .get(operand_index as usize - 1)
            .copied()
            .unwrap_or(1.0)
    }
}

pub(crate) fn homogeneous_tangent(input_slice: &[f32], operand_index: u16) -> f32 {
    if operand_index == 0 {
        0.0
    } else {
        input_slice
            .get(operand_index as usize - 1)
            .copied()
            .unwrap_or(0.0)
    }
}

fn operand_value(operand_index: u16, input_slice: &[f32]) -> Option<f32> {
    if operand_index == UNUSED_OPERAND {
        return None;
    }
    Some(input_slice[operand_index as usize])
}

fn operand_tangent(operand_index: u16, tangent_input_slice: &[f32]) -> Option<f32> {
    if operand_index == UNUSED_OPERAND {
        return None;
    }
    Some(tangent_input_slice[operand_index as usize])
}

fn required_operand(operand_value: Option<f32>) -> f32 {
    operand_value.expect("validated generic operation missing required operand")
}

fn divide(num: f32, den: f32) -> f32 {
    num / den
}
