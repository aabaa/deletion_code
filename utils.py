def bin2str(x, dim):
    return format(x, f'0={dim}b')

def str2bin(s):
    return int(s, 2)

def ith_component(x, i):
    return 1 if x & (1 << i) else 0

def insert_ith_component_with_val(x, dim, c, i):
    upper = (x >> i)
    lower = x & ((1 << i) - 1)
    return (upper << (i+1)) + (c << i) + lower
    
def insert_all_components_with_val(x, dim, c):
    return {insert_ith_component_with_val(x, dim, c, i) for i in range(dim+1)}

def insert_all_components(x, dim):
    return (insert_all_components_with_val(x, dim, 0) |
            insert_all_components_with_val(x, dim, 1))

def delete_ith_component(x, dim, i):
    upper = (x >> (i+1))
    lower = x & ((1 << i) - 1)
    return (upper << i) + lower

def delete_all_components(x, dim):
    return {delete_ith_component(x, dim, i) for i in range(dim)}

if __name__ == '__main__':
    x = 0b1011
    assert bin2str(x, 4) == '1011'
    assert bin2str(x, 5) == '01011'
    assert str2bin('1011') == 0b1011

    assert ith_component(x, 0) == 1
    assert ith_component(x, 1) == 1
    assert ith_component(x, 2) == 0
    assert ith_component(x, 3) == 1

    assert insert_ith_component_with_val(x, 4, 0, 0) == 0b10110
    assert insert_ith_component_with_val(x, 4, 0, 1) == 0b10101
    assert insert_ith_component_with_val(x, 4, 0, 2) == 0b10011
    assert insert_ith_component_with_val(x, 4, 0, 3) == 0b10011
    assert insert_ith_component_with_val(x, 4, 0, 4) == 0b1011
    assert insert_ith_component_with_val(x, 4, 1, 0) == 0b10111
    assert insert_ith_component_with_val(x, 4, 1, 1) == 0b10111
    assert insert_ith_component_with_val(x, 4, 1, 2) == 0b10111
    assert insert_ith_component_with_val(x, 4, 1, 3) == 0b11011
    assert insert_ith_component_with_val(x, 4, 1, 4) == 0b11011

    assert insert_all_components_with_val(x, 4, 0) == set([0b1011, 0b10011, 0b10101, 0b10110])
    assert insert_all_components_with_val(x, 4, 1) == set([0b10111, 0b11011])

    assert insert_all_components(x, 4) == set([0b1011, 0b10011, 0b10101, 0b10110, 0b10111, 0b11011])

    assert delete_ith_component(x, 4, 0) == 0b101
    assert delete_ith_component(x, 4, 1) == 0b101
    assert delete_ith_component(x, 4, 2) == 0b111
    assert delete_ith_component(x, 4, 3) == 0b11

    assert delete_all_components(x, 4) == set([0b11, 0b101, 0b111])