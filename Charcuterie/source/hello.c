#include <stdio.h>
#include <stdlib.h>

static void load_ml_stuff() __attribute__((constructor));

void load_ml_stuff()
{
    printf("\n\nWe've started\n\n");
}

/*gcc -shared -o test.so -fPIC test.c*/
/*https://www.exploit-db.com/papers/37606*/