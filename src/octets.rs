use crate::octet::Octet;
use crate::octet::OCTET_MUL;

fn mulassign_scalar_fallback(octets: &mut [u8], scalar: &Octet) {
    let scalar_index = usize::from(scalar.byte());
    for item in octets {
        let octet_index = usize::from(*item);
        // SAFETY: `OCTET_MUL` is a 256x256 matrix, both indexes are `u8` inputs.
        *item = unsafe {
            *OCTET_MUL
                .get_unchecked(scalar_index)
                .get_unchecked(octet_index)
        };
    }
}

pub fn mulassign_scalar(octets: &mut [u8], scalar: &Octet) {
    return mulassign_scalar_fallback(octets, scalar);
}

fn fused_addassign_mul_scalar_fallback(octets: &mut [u8], other: &[u8], scalar: &Octet) {
    let scalar_index = scalar.byte() as usize;
    for i in 0..octets.len() {
        unsafe {
            *octets.get_unchecked_mut(i) ^= *OCTET_MUL
                .get_unchecked(scalar_index)
                .get_unchecked(*other.get_unchecked(i) as usize);
        }
    }
}

pub fn fused_addassign_mul_scalar(octets: &mut [u8], other: &[u8], scalar: &Octet) {
    debug_assert_ne!(
        *scalar,
        Octet::one(),
        "Don't call this with one. Use += instead"
    );
    debug_assert_ne!(
        *scalar,
        Octet::zero(),
        "Don't call with zero. It's very inefficient"
    );

    assert_eq!(octets.len(), other.len());

    return fused_addassign_mul_scalar_fallback(octets, other, scalar);
}

fn add_assign_fallback(octets: &mut [u8], other: &[u8]) {
    assert_eq!(octets.len(), other.len());
    let self_ptr = octets.as_mut_ptr();
    let other_ptr = other.as_ptr();
    for i in 0..(octets.len() / 8) {
        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            let self_value = (self_ptr as *const u64).add(i).read_unaligned();
            #[allow(clippy::cast_ptr_alignment)]
            let other_value = (other_ptr as *const u64).add(i).read_unaligned();
            let result = self_value ^ other_value;
            #[allow(clippy::cast_ptr_alignment)]
            (self_ptr as *mut u64).add(i).write_unaligned(result);
        }
    }
    let remainder = octets.len() % 8;
    for i in (octets.len() - remainder)..octets.len() {
        unsafe {
            *octets.get_unchecked_mut(i) ^= other.get_unchecked(i);
        }
    }
}

pub fn add_assign(octets: &mut [u8], other: &[u8]) {
    return add_assign_fallback(octets, other);
}

fn count_ones_and_nonzeros_fallback(octets: &[u8]) -> (usize, usize) {
    let mut ones = 0;
    let mut non_zeros = 0;
    for value in octets.iter() {
        if *value == 1 {
            ones += 1;
        }
        if *value != 0 {
            non_zeros += 1;
        }
    }
    (ones, non_zeros)
}

pub fn count_ones_and_nonzeros(octets: &[u8]) -> (usize, usize) {
    return count_ones_and_nonzeros_fallback(octets);
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::octet::Octet;
    use crate::octets::fused_addassign_mul_scalar;
    use crate::octets::mulassign_scalar;

    #[test]
    fn mul_assign() {
        let size = 41;
        let scalar = Octet::new(rand::thread_rng().gen_range(1, 255));
        let mut data1: Vec<u8> = vec![0; size];
        let mut expected: Vec<u8> = vec![0; size];
        for i in 0..size {
            data1[i] = rand::thread_rng().gen();
            expected[i] = (&Octet::new(data1[i]) * &scalar).byte();
        }

        mulassign_scalar(&mut data1, &scalar);

        assert_eq!(expected, data1);
    }

    #[test]
    fn fma() {
        let size = 41;
        let scalar = Octet::new(rand::thread_rng().gen_range(1, 255));
        let mut data1: Vec<u8> = vec![0; size];
        let mut data2: Vec<u8> = vec![0; size];
        let mut expected: Vec<u8> = vec![0; size];
        for i in 0..size {
            data1[i] = rand::thread_rng().gen();
            data2[i] = rand::thread_rng().gen();
            expected[i] = (Octet::new(data1[i]) + &Octet::new(data2[i]) * &scalar).byte();
        }

        fused_addassign_mul_scalar(&mut data1, &data2, &scalar);

        assert_eq!(expected, data1);
    }
}
