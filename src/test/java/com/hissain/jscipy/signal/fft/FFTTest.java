package com.hissain.jscipy.signal.fft;

import com.hissain.jscipy.TestMetrics;
import com.hissain.jscipy.signal.JComplex;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class FFTTest {

    private double[] load1DArray(String filePath) throws IOException {
        List<Double> list = new ArrayList<>();
        File file = new File(filePath);
        if (!file.exists()) {
            file = new File("datasets/fft/" + filePath);
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    list.add(Double.parseDouble(line));
                }
            }
        }
        double[] array = new double[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    private JComplex[] loadComplexArray(String filePath) throws IOException {
        List<JComplex> list = new ArrayList<>();
        File file = new File(filePath);
        if (!file.exists()) {
            file = new File("datasets/fft/" + filePath);
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    String[] parts = line.split("\\s+");
                    double real = Double.parseDouble(parts[0]);
                    double imag = Double.parseDouble(parts[1]);
                    list.add(new JComplex(real, imag));
                }
            }
        }
        return list.toArray(new JComplex[0]);
    }

    private double calculateRMSE(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length, "Array lengths must match");
        double sum = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double diff = expected[i] - actual[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum / expected.length);
    }

    private double calculateComplexRMSE(JComplex[] expected, JComplex[] actual) {
        assertEquals(expected.length, actual.length, "Complex array lengths must match");
        double sum = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double diffReal = expected[i].getReal() - actual[i].getReal();
            double diffImag = expected[i].getImaginary() - actual[i].getImaginary();
            sum += diffReal * diffReal + diffImag * diffImag;
        }
        return Math.sqrt(sum / expected.length);
    }

    @Test
    public void testFFT_double() {
        FFT fftObject = new FFT();
        double[] input = { 1, 2, 3, 4 }; // Power of 2
        JComplex[] output = fftObject.fft(input);
        assertNotNull(output);
        assertEquals(4, output.length);

        double[] inputNon2 = { 1, 2, 3 }; // Non power of 2
        JComplex[] outputNon2 = fftObject.fft(inputNon2);
        assertNotNull(outputNon2);
        assertEquals(3, outputNon2.length);
    }

    @Test
    public void testFFT_complex() {
        FFT fftObject = new FFT();
        JComplex[] input = { new JComplex(1, 0), new JComplex(2, 0) };
        JComplex[] output = fftObject.fft(input);
        assertNotNull(output);

        // Non power of 2
        JComplex[] inputNon2 = { new JComplex(1, 0), new JComplex(2, 0), new JComplex(3, 0) };
        JComplex[] outputNon2 = fftObject.fft(inputNon2);
        assertNotNull(outputNon2);
    }

    @Test
    public void testIFFT() {
        FFT fftObject = new FFT();
        JComplex[] input = { new JComplex(1, 0), new JComplex(2, 0) };
        JComplex[] output = fftObject.ifft(input);
        assertNotNull(output);

        // Non power of 2
        JComplex[] inputNon2 = { new JComplex(1, 0), new JComplex(2, 0), new JComplex(3, 0) };
        JComplex[] outputNon2 = fftObject.ifft(inputNon2);
        assertNotNull(outputNon2);
    }

    @Test
    public void testFFT2() {
        FFT fftObject = new FFT();
        double[][] input = { { 1, 2 }, { 3, 4 } };
        JComplex[][] output = fftObject.fft2(input);
        assertNotNull(output);
        assertEquals(2, output.length);
        assertEquals(2, output[0].length);

        JComplex[][] ifftOutput = fftObject.ifft2(output);
        assertNotNull(ifftOutput);
    }

    @Test
    public void testRFFT() {
        FFT fftObject = new FFT();
        double[] input = { 1, 2, 3, 4 };
        JComplex[] output = fftObject.rfft(input);
        assertNotNull(output);
        // rfft returns n/2 + 1 elements
        assertEquals(3, output.length);

        double[] recon = fftObject.irfft(output, 4);
        assertNotNull(recon);
        assertEquals(4, recon.length);
    }

    @Test
    public void testSTFT() {
        FFT fftObject = new FFT();
        double[] input = new double[100];
        for (int i = 0; i < 100; i++)
            input[i] = Math.sin(0.1 * i);

        // Default STFT
        JComplex[][] stft = fftObject.stft(input);
        assertNotNull(stft);

        // Custom STFT
        JComplex[][] stftCustom = fftObject.stft(input, 32, 16, 64, null, "zeros", true);
        assertNotNull(stftCustom);

        // ISTFT
        double[] output = fftObject.istft(stft);
        assertNotNull(output);
    }

    @Test
    public void testEmptyAndSingleElement() {
        FFT fft = new FFT();
        assertEquals(0, fft.fft(new double[0]).length);
        assertEquals(0, fft.fft(new JComplex[0]).length);
        assertEquals(0, fft.ifft(new JComplex[0]).length);
        assertEquals(0, fft.rfft(new double[0]).length);
        assertEquals(0, fft.irfft(new JComplex[0], 0).length);

        // Length 1
        double[] x1 = { 5.5 };
        JComplex[] out1 = fft.fft(x1);
        assertEquals(1, out1.length);
        assertEquals(5.5, out1[0].getReal(), 1e-12);
        assertEquals(0.0, out1[0].getImaginary(), 1e-12);

        JComplex[] ifft1 = fft.ifft(out1);
        assertEquals(5.5, ifft1[0].getReal(), 1e-12);

        JComplex[] rfft1 = fft.rfft(x1);
        assertEquals(1, rfft1.length);
        double[] irfft1 = fft.irfft(rfft1, 1);
        assertEquals(5.5, irfft1[0], 1e-12);
    }

    @Test
    public void testSciPyDatasetsMatch() throws IOException {
        FFT fft = new FFT();
        int[] testIds = { 1, 2, 3, 4, 5 };
        String[] descriptions = {
            "Power-of-2 (64)",
            "Power-of-2 (128)",
            "Composite Non-Power-of-2 (100)",
            "Prime Non-Power-of-2 (97)",
            "Large Non-Power-of-2 (1000)"
        };

        for (int i = 0; i < testIds.length; i++) {
            int id = testIds[i];
            String desc = descriptions[i];

            double[] input = load1DArray("datasets/fft/fft_input_" + id + ".txt");
            JComplex[] expectedFft = loadComplexArray("datasets/fft/fft_output_" + id + ".txt");
            JComplex[] expectedRfft = loadComplexArray("datasets/fft/rfft_output_" + id + ".txt");
            JComplex[] expectedIfft = loadComplexArray("datasets/fft/ifft_output_" + id + ".txt");
            double[] expectedIrfft = load1DArray("datasets/fft/irfft_output_" + id + ".txt");

            // FFT
            JComplex[] actualFft = fft.fft(input);
            double rmseFft = calculateComplexRMSE(expectedFft, actualFft);
            TestMetrics.log("FFT", "FFT_" + desc, rmseFft);
            assertTrue(rmseFft < 1e-10, "FFT failed for " + desc + " with RMSE " + rmseFft);

            // RFFT
            JComplex[] actualRfft = fft.rfft(input);
            double rmseRfft = calculateComplexRMSE(expectedRfft, actualRfft);
            TestMetrics.log("FFT", "RFFT_" + desc, rmseRfft);
            assertTrue(rmseRfft < 1e-10, "RFFT failed for " + desc + " with RMSE " + rmseRfft);

            // IFFT
            JComplex[] actualIfft = fft.ifft(actualFft);
            double rmseIfft = calculateComplexRMSE(expectedIfft, actualIfft);
            TestMetrics.log("FFT", "IFFT_" + desc, rmseIfft);
            assertTrue(rmseIfft < 1e-10, "IFFT failed for " + desc + " with RMSE " + rmseIfft);

            // IRFFT
            double[] actualIrfft = fft.irfft(actualRfft, input.length);
            double rmseIrfft = calculateRMSE(expectedIrfft, actualIrfft);
            TestMetrics.log("FFT", "IRFFT_" + desc, rmseIrfft);
            assertTrue(rmseIrfft < 1e-10, "IRFFT failed for " + desc + " with RMSE " + rmseIrfft);
        }
    }

    @Test
    public void testNonPowerOf2Performance() {
        // Test that a large non-power-of-2 transform runs fast (O(N log N))
        // Naive DFT for N=10000 takes seconds; JTransforms should complete in milliseconds
        int n = 10007; // Prime length!
        double[] input = new double[n];
        for (int i = 0; i < n; i++) {
            input[i] = Math.sin(2 * Math.PI * 50 * i / n);
        }

        FFT fft = new FFT();
        long startTime = System.currentTimeMillis();
        JComplex[] result = fft.fft(input);
        JComplex[] reconstructed = fft.ifft(result);
        long elapsed = System.currentTimeMillis() - startTime;

        assertNotNull(result);
        assertEquals(n, result.length);
        assertEquals(input[0], reconstructed[0].getReal(), 1e-8);
        System.out.println("Non-power-of-2 prime FFT+IFFT of size " + n + " completed in " + elapsed + " ms");
        assertTrue(elapsed < 2000, "FFT of size " + n + " should be fast (O(N log N)), took " + elapsed + " ms");
    }
}
