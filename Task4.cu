

/*
 Author: Oladipupo Roland Familua
 Description:
 This program applies the Sobel edge detection operator to a PNG image
 using CUDA for parallel processing on the GPU. The input image is
 loaded on the CPU, processed in parallel on the GPU, and then saved
 back to disk as a new image.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h"

/*
 CUDA Kernel: Sobel
 ------------------
 This kernel performs Sobel edge detection on an RGBA image.
 Each thread processes exactly one pixel, computing the
 gradient magnitude for the Red, Green, and Blue channels.
*/
__global__ void Sobel(
    unsigned char *inputImage,   // Input RGBA image on GPU
    unsigned char *outputImage,  // Output RGBA image on GPU
    int width,                   // Image width
    int height                   // Image height
) {
    /* Thread and block mapping:
       - x coordinate comes from thread index
       - y coordinate comes from block index
    */
    int x = threadIdx.x;
    int y = blockIdx.x;

    /* Boundary check to prevent illegal memory access */
    if (x >= width || y >= height) return;

    /* Sobel filter kernels for horizontal (Gx) and vertical (Gy) gradients */
    int Gx[3][3] = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };
    int Gy[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    /* Gradient accumulators for each color channel */
    float sumRx = 0, sumRy = 0;
    float sumGx = 0, sumGy = 0;
    float sumBx = 0, sumBy = 0;

    /*
     Perform 3x3 convolution around the current pixel.
     Boundary pixels are clamped to the image edges.
    */
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            int curX = x + j;
            int curY = y + i;

            /* Clamp coordinates to image boundaries */
            if (curX < 0) curX = 0;
            else if (curX >= width) curX = width - 1;

            if (curY < 0) curY = 0;
            else if (curY >= height) curY = height - 1;

            /* Convert 2D pixel coordinate to 1D RGBA index */
            int offset = (curY * width + curX) * 4;

            int weightX = Gx[i + 1][j + 1];
            int weightY = Gy[i + 1][j + 1];

            /* Accumulate Red channel gradients */
            sumRx += inputImage[offset]     * weightX;
            sumRy += inputImage[offset]     * weightY;

            /* Accumulate Green channel gradients */
            sumGx += inputImage[offset + 1] * weightX;
            sumGy += inputImage[offset + 1] * weightY;

            /* Accumulate Blue channel gradients */
            sumBx += inputImage[offset + 2] * weightX;
            sumBy += inputImage[offset + 2] * weightY;
        }
    }

    /* Compute gradient magnitude for each channel */
    int magR = (int) sqrtf(sumRx * sumRx + sumRy * sumRy);
    int magG = (int) sqrtf(sumGx * sumGx + sumGy * sumGy);
    int magB = (int) sqrtf(sumBx * sumBx + sumBy * sumBy);

    /* Clamp values to valid 8-bit color range */
    if (magR > 255) magR = 255;
    if (magG > 255) magG = 255;
    if (magB > 255) magB = 255;

    /* Write result pixel to output image */
    int pixelIdx = (y * width + x) * 4;

    outputImage[pixelIdx]     = (unsigned char) magR;
    outputImage[pixelIdx + 1] = (unsigned char) magG;
    outputImage[pixelIdx + 2] = (unsigned char) magB;

    /* Alpha channel set to fully opaque */
    outputImage[pixelIdx + 3] = 255;
}

/*
 Main Function
 -------------
 Handles input/output, memory allocation, kernel execution,
 and saving the processed image.
*/
int main(int argc, char **argv) {

    /* Validate command-line arguments */
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    char *newFilename = argv[2];

    unsigned char *cpuImage;
    unsigned int width, height;
    unsigned int error;

    /* Load PNG image into CPU memory as RGBA */
    error = lodepng_decode32_file(&cpuImage, &width, &height, filename);
    if (error) {
        printf("Decoder error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    printf("Processing %s (%d x %d)...\n", filename, width, height);

    /* Allocate memory sizes */
    int arraySize = width * height * 4;
    int memorySize = arraySize * sizeof(unsigned char);

    unsigned char *gpuInput;
    unsigned char *gpuOutput;
    unsigned char *cpuOutput = (unsigned char *) malloc(memorySize);

    /* Allocate GPU memory */
    cudaMalloc((void**)&gpuInput, memorySize);
    cudaMalloc((void**)&gpuOutput, memorySize);

    /* Copy input image from CPU to GPU */
    cudaMemcpy(gpuInput, cpuImage, memorySize, cudaMemcpyHostToDevice);

    /*
     Launch Sobel kernel:
     - One block per image row
     - One thread per image column
    */
    Sobel<<< dim3(height, 1, 1), dim3(width, 1, 1) >>>(
        gpuInput, gpuOutput, width, height
    );

    /* Synchronize CPU and GPU */
    cudaThreadSynchronize();

    /* Copy processed image back to CPU */
    cudaMemcpy(cpuOutput, gpuOutput, memorySize, cudaMemcpyDeviceToHost);

    /* Save output image */
    error = lodepng_encode32_file(newFilename, cpuOutput, width, height);
    if (error) {
        printf("Encoder error %u: %s\n", error, lodepng_error_text(error));
    } else {
        printf("Success! Saved to %s\n", newFilename);
    }

    /* Free allocated memory */
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    free(cpuImage);
    free(cpuOutput);

    return 0;
}
