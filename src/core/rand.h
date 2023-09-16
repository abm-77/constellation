#include <stdlib.h>
#include <core/language_layer.h>

i32 irand() {
    return (i32) rand();

}

i32 irand_range(i32 min, i32 max) {
    int diff = max - min;
    return (i32) (((f64)(diff+1)/ RAND_MAX) * rand() + min);
}
