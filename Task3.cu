#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

/*
 * Author: Oladipupo Roland Familua
 *
 * This program uses CUDA to brute-force decrypt encrypted passwords.
 * Each encrypted password is assumed to be generated from a 4-character
 * raw password consisting of:
 *   - 2 lowercase letters (a–z)
 *   - 2 digits (0–9)
 *
 * The program launches one CUDA block per encrypted password and
 * distributes the brute-force search across threads within that block.
 */

#define RAW_PASS_LEN 4     // Length of the original (raw) password
#define ENC_PASS_LEN 10    // Length of the encrypted password

// ---------------- DEVICE ENCRYPTION FUNCTION ----------------
/*
 * cudaCryptDevice
 * ----------------
 * This device function performs the same encryption logic that was
 * originally used to generate the encrypted passwords.
 *
 * It takes a 4-character raw password and produces a 10-character
 * encrypted output by applying fixed arithmetic transformations.
 *
 * Character wrapping is applied:
 *  - First 6 characters are wrapped within 'a'–'z'
 *  - Last 4 characters are wrapped within '0'–'9'
 *
 * This function is called repeatedly inside the kernel to test
 * candidate passwords.
 */
__device__ void cudaCryptDevice(char *raw, char *out) {
    out[0] = raw[0] + 2;
    out[1] = raw[0] - 2;
    out[2] = raw[0] + 1;

    out[3] = raw[1] + 3;
    out[4] = raw[1] - 3;
    out[5] = raw[1] - 1;

    out[6] = raw[2] + 2;
    out[7] = raw[2] - 2;

    out[8] = raw[3] + 4;
    out[9] = raw[3] - 4;

    // Wrap characters to ensure valid ranges
    for (int i = 0; i < 10; i++) {
        if (i < 6) {
            // Wrap lowercase letters
            if (out[i] > 'z') out[i] = (out[i] - 'z') + 'a';
            else if (out[i] < 'a') out[i] = ('a' - out[i]) + 'a';
        } else {
            // Wrap digits
            if (out[i] > '9') out[i] = (out[i] - '9') + '0';
            else if (out[i] < '0') out[i] = ('0' - out[i]) + '0';
        }
    }
}

// ------------------- CUDA KERNEL ----------------------------
/*
 * crackKernel
 * ------------
 * Each CUDA block is responsible for cracking one encrypted password.
 * Threads within the block split the total brute-force search space.
 *
 * The kernel:
 *  - Loads the encrypted password for its block
 *  - Generates candidate raw passwords
 *  - Encrypts each candidate using cudaCryptDevice
 *  - Compares the result to the target encrypted password
 *  - Writes the matching raw password to global memory
 */
__global__ void crackKernel(char *encryptedList, char *outputRaw, int count) {
    int index = blockIdx.x;  // One block per encrypted password

    // Prevent out-of-bounds access
    if (index >= count) return;

    char target[ENC_PASS_LEN + 1];
    char genEnc[ENC_PASS_LEN + 1];
    char candidate[RAW_PASS_LEN + 1];

    // Copy the encrypted password for this block into local memory
    for (int i = 0; i < ENC_PASS_LEN; i++)
        target[i] = encryptedList[index * ENC_PASS_LEN + i];
    target[ENC_PASS_LEN] = '\0';

    // Thread identification within the block
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    // Total combinations: 26 letters × 26 letters × 10 digits × 10 digits
    int totalCombos = 26 * 26 * 10 * 10;

    // Each thread processes a strided subset of the search space
    for (int t = tid; t < totalCombos; t += totalThreads) {
        int x = t;

        // Decode the integer into password components
        int d2 = x % 10; x /= 10;
        int d1 = x % 10; x /= 10;
        int l2 = x % 26; x /= 26;
        int l1 = x % 26;

        // Construct candidate raw password
        candidate[0] = 'a' + l1;
        candidate[1] = 'a' + l2;
        candidate[2] = '0' + d1;
        candidate[3] = '0' + d2;
        candidate[4] = '\0';

        // Encrypt candidate password
        cudaCryptDevice(candidate, genEnc);

        // Compare encrypted candidate with target
        bool match = true;
        for (int j = 0; j < ENC_PASS_LEN; j++) {
            if (genEnc[j] != target[j]) {
                match = false;
                break;
            }
        }

        // If a match is found, store the raw password and exit
        if (match) {
            for (int k = 0; k < RAW_PASS_LEN; k++)
                outputRaw[index * RAW_PASS_LEN + k] = candidate[k];
            return;
        }
    }
}

// ------------------------- HOST CODE -------------------------
/*
 * main
 * ----
 * Host-side code responsible for:
 *  - Reading encrypted passwords from a file
 *  - Allocating host and device memory
 *  - Launching the CUDA kernel
 *  - Retrieving and writing decrypted passwords to a file
 */
int main(int argc, char *argv[]) {
    // Validate command-line arguments
    if (argc != 3) {
        printf("Usage: %s <file> <threadsPerBlock>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int threads = atoi(argv[2]);

    // Validate thread count
    if (threads <= 0 || threads > 1024) {
        printf("Invalid number of threads (1-1024 allowed)\n");
        return 1;
    }

    // Open encrypted password file
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Cannot open file %s\n", filename);
        return 1;
    }

    // Count number of encrypted passwords (lines in file)
    int count = 0;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), fp)) count++;
    rewind(fp);

    // Allocate host memory
    char *h_enc = (char *)malloc(count * ENC_PASS_LEN);
    char *h_raw = (char *)malloc(count * RAW_PASS_LEN);

    // Read encrypted passwords into host memory
    for (int i = 0; i < count; i++) {
        fgets(buffer, sizeof(buffer), fp);
        buffer[strcspn(buffer, "\n")] = 0;
        memcpy(&h_enc[i * ENC_PASS_LEN], buffer, ENC_PASS_LEN);
    }
    fclose(fp);

    // Allocate device memory
    char *d_enc, *d_raw;
    cudaMalloc(&d_enc, count * ENC_PASS_LEN);
    cudaMalloc(&d_raw, count * RAW_PASS_LEN);

    // Copy encrypted passwords to device
    cudaMemcpy(d_enc, h_enc, count * ENC_PASS_LEN, cudaMemcpyHostToDevice);

    // Launch kernel: one block per encrypted password
    crackKernel<<<count, threads>>>(d_enc, d_raw, count);
    cudaDeviceSynchronize();

    // Copy decrypted passwords back to host
    cudaMemcpy(h_raw, d_raw, count * RAW_PASS_LEN, cudaMemcpyDeviceToHost);

    // Write decrypted passwords to output file
    FILE *out = fopen("decrypted.txt", "w");
    for (int i = 0; i < count; i++) {
        fprintf(out, "%c%c%c%c\n",
                h_raw[i * RAW_PASS_LEN],
                h_raw[i * RAW_PASS_LEN + 1],
                h_raw[i * RAW_PASS_LEN + 2],
                h_raw[i * RAW_PASS_LEN + 3]);
    }
    fclose(out);

    // Free device and host memory
    cudaFree(d_enc);
    cudaFree(d_raw);
    free(h_enc);
    free(h_raw);

    printf("Complete. Output written to decrypted.txt\n");

    return 0;
}
