# qr

Usage:
```bash
make
./test_qr 27648 128 2
```

# tsqr

Usage:
```bash
make
./test_tsqr 27648 32 2
```
## Constraints:

- m >= n
- (m % BLOCK_SIZE) % n == 0


## Tested Parameters:

float:
|      | n = 8 | n = 16 | n = 32 | n = 64 |
| :--: | :--- | :------ | :----- | :----- |
| H100 | / | / | TODO | TODO |
| A800 | / | / | #define NUM_SM 108<br/>#define BLOCK_SIZE 256<br/>max grid size = 27648<br />max supported size = 23887872 | / |
| 4090 | / | / | #define NUM_SM 100<br/>#define BLOCK_SIZE 128<br/>max grid size = 12800<br />max supported size = 5120000 | / |

double:

|      | n = 8 | n = 16 | n = 32 | n = 64 |
| :--: | :--- | :------ | :----- | :----- |
| H100 | / | / | TODO | TODO |
| A800 | / | / | #define NUM_SM 108<br/>#define BLOCK_SIZE 128<br/>max grid size = 13824<br />max supported size = 5971968 | / |
| 4090 | / | / | / | / |

`NUM_SM` and `NUM_SM` are located at the top of [kernelQR.h](kernelQR.h). max supported size is the maximum input height (m).
