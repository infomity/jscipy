package com.hissain.jscipy.signal.fft;

import com.hissain.jscipy.signal.JComplex;
import com.hissain.jscipy.signal.Windows;
import org.jtransforms.fft.DoubleFFT_1D;
import org.jtransforms.fft.DoubleFFT_2D;

import java.util.Arrays;

/**
 * A class for performing Fast Fourier Transforms (FFT) and Short-Time Fourier
 * Transforms (STFT).
 * This class uses the JTransforms library for efficient O(N log N) FFT calculations
 * across arbitrary sizes (powers of 2, composite, and prime lengths).
 * <p>
 * <img src=
 * "https://raw.githubusercontent.com/hissain/jscipy/main/python/figs/fft/fft_comparison_1_light.png"
 * alt="FFT Comparison" style="width: 600px; max-width: 90%; display: block; margin: 0 auto;">
 */
public class FFT {

    /**
     * Constructs a new FFT object.
     */
    public FFT() {
    }

    /**
     * Computes the forward FFT of a real-valued signal.
     * 
     * @param input The input signal.
     * @return The complex-valued FFT of the signal as an array of {@link JComplex}
     *         objects.
     */
    public JComplex[] fft(double[] input) {
        int n = input.length;
        if (n == 0) {
            return new JComplex[0];
        }
        DoubleFFT_1D fft = new DoubleFFT_1D(n);
        double[] data = new double[2 * n];
        System.arraycopy(input, 0, data, 0, n);
        fft.realForwardFull(data);
        JComplex[] output = new JComplex[n];
        for (int i = 0; i < n; i++) {
            output[i] = new JComplex(data[2 * i], data[2 * i + 1]);
        }
        return output;
    }

    /**
     * Computes the forward FFT of a complex-valued signal.
     * 
     * @param input The complex-valued input signal.
     * @return The complex-valued FFT of the signal.
     */
    public JComplex[] fft(JComplex[] input) {
        int n = input.length;
        if (n == 0) {
            return new JComplex[0];
        }
        DoubleFFT_1D fft = new DoubleFFT_1D(n);
        double[] data = new double[2 * n];
        for (int i = 0; i < n; i++) {
            data[2 * i] = input[i].getReal();
            data[2 * i + 1] = input[i].getImaginary();
        }
        fft.complexForward(data);
        JComplex[] output = new JComplex[n];
        for (int i = 0; i < n; i++) {
            output[i] = new JComplex(data[2 * i], data[2 * i + 1]);
        }
        return output;
    }

    /**
     * Computes the inverse FFT of a complex-valued signal.
     * 
     * @param input The complex-valued input signal.
     * @return The complex-valued inverse FFT of the signal as an array of
     *         {@link JComplex} objects.
     */
    public JComplex[] ifft(JComplex[] input) {
        int n = input.length;
        if (n == 0) {
            return new JComplex[0];
        }
        DoubleFFT_1D fft = new DoubleFFT_1D(n);
        double[] data = new double[2 * n];
        for (int i = 0; i < n; i++) {
            data[2 * i] = input[i].getReal();
            data[2 * i + 1] = input[i].getImaginary();
        }
        fft.complexInverse(data, true);
        JComplex[] output = new JComplex[n];
        for (int i = 0; i < n; i++) {
            output[i] = new JComplex(data[2 * i], data[2 * i + 1]);
        }
        return output;
    }

    /**
     * Computes the 2D FFT of a real-valued 2D array.
     * <p>
     * <img src=
     * "https://raw.githubusercontent.com/hissain/jscipy/main/python/figs/fft2/fft2_comparison_forward_light.png"
     * alt="FFT2 Comparison" style="width: 600px; max-width: 90%; display: block; margin: 0 auto;">
     * 
     * @param input The 2D input array [rows][cols].
     * @return The 2D complex-valued FFT.
     */
    public JComplex[][] fft2(double[][] input) {
        int rows = input.length;
        if (rows == 0) return new JComplex[0][0];
        int cols = input[0].length;
        if (cols == 0) return new JComplex[rows][0];

        DoubleFFT_2D fft2d = new DoubleFFT_2D(rows, cols);
        double[][] data = new double[rows][2 * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                data[r][2 * c] = input[r][c];
                data[r][2 * c + 1] = 0.0;
            }
        }
        fft2d.complexForward(data);

        JComplex[][] output = new JComplex[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                output[r][c] = new JComplex(data[r][2 * c], data[r][2 * c + 1]);
            }
        }
        return output;
    }

    /**
     * Computes the 2D FFT of a complex-valued 2D array.
     * 
     * @param input The 2D complex input array [rows][cols].
     * @return The 2D complex-valued FFT.
     */
    public JComplex[][] fft2(JComplex[][] input) {
        int rows = input.length;
        if (rows == 0) return new JComplex[0][0];
        int cols = input[0].length;
        if (cols == 0) return new JComplex[rows][0];

        DoubleFFT_2D fft2d = new DoubleFFT_2D(rows, cols);
        double[][] data = new double[rows][2 * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                data[r][2 * c] = input[r][c].getReal();
                data[r][2 * c + 1] = input[r][c].getImaginary();
            }
        }
        fft2d.complexForward(data);

        JComplex[][] output = new JComplex[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                output[r][c] = new JComplex(data[r][2 * c], data[r][2 * c + 1]);
            }
        }
        return output;
    }

    /**
     * Computes the inverse 2D FFT.
     * 
     * @param input The 2D complex spectrum [rows][cols].
     * @return The 2D complex spatial domain signal.
     */
    public JComplex[][] ifft2(JComplex[][] input) {
        int rows = input.length;
        if (rows == 0) return new JComplex[0][0];
        int cols = input[0].length;
        if (cols == 0) return new JComplex[rows][0];

        DoubleFFT_2D fft2d = new DoubleFFT_2D(rows, cols);
        double[][] data = new double[rows][2 * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                data[r][2 * c] = input[r][c].getReal();
                data[r][2 * c + 1] = input[r][c].getImaginary();
            }
        }
        fft2d.complexInverse(data, true);

        JComplex[][] output = new JComplex[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                output[r][c] = new JComplex(data[r][2 * c], data[r][2 * c + 1]);
            }
        }
        return output;
    }

    /**
     * Computes the forward FFT of a real-valued signal and returns the positive
     * frequency components.
     * 
     * @param input The input signal.
     * @return The positive frequency components of the FFT as an array of
     *         {@link JComplex} objects.
     */
    public JComplex[] rfft(double[] input) {
        int n = input.length;
        if (n == 0) {
            return new JComplex[0];
        }
        JComplex[] fftResult = fft(input);
        int resultSize = n / 2 + 1;
        return Arrays.copyOf(fftResult, resultSize);
    }

    /**
     * Computes the inverse FFT of a real-valued signal.
     * 
     * @param input The complex-valued input signal.
     * @param n     The length of the original signal.
     * @return The real-valued inverse FFT of the signal.
     */
    public double[] irfft(JComplex[] input, int n) {
        if (n == 0) {
            return new double[0];
        }
        // Reconstruct the full complex spectrum from the RFFT output
        JComplex[] fullSpectrum = new JComplex[n];
        fullSpectrum[0] = input[0]; // DC component

        if (n % 2 == 0) {
            for (int i = 1; i < n / 2; i++) {
                fullSpectrum[i] = input[i];
                fullSpectrum[n - i] = input[i].conjugate();
            }
            if (n > 1) {
                fullSpectrum[n / 2] = input[n / 2];
            }
        } else {
            for (int i = 1; i <= n / 2; i++) {
                fullSpectrum[i] = input[i];
                fullSpectrum[n - i] = input[i].conjugate();
            }
        }

        JComplex[] ifftResult = ifft(fullSpectrum);
        double[] realResult = new double[n];
        for (int i = 0; i < n; i++) {
            realResult[i] = ifftResult[i].getReal();
        }
        return realResult;
    }

    /**
     * Computes the Short-Time Fourier Transform (STFT) with default parameters.
     * Uses nperseg=256, noverlap=128, Hann window, and zero-padding.
     * 
     * @param x The input signal.
     * @return A 2D array of complex values [frequency bins][time frames]
     *         representing the STFT.
     */
    public JComplex[][] stft(double[] x) {
        return stft(x, -1, -1, -1, null, "zeros", true);
    }

    /**
     * Computes the Inverse Short-Time Fourier Transform (ISTFT) with default
     * parameters.
     * Uses nperseg=256, noverlap=128, Hann window, and assumes zero-padding was
     * used.
     * 
     * @param stftMatrix The STFT matrix [frequency bins][time frames].
     * @return The reconstructed time-domain signal.
     */
    public double[] istft(JComplex[][] stftMatrix) {
        return istft(stftMatrix, -1, -1, -1, null, "zeros", -1);
    }

    /**
     * Computes the Short-Time Fourier Transform (STFT) of a signal.
     * This implementation matches the behavior of scipy.signal.stft with default
     * parameters.
     * 
     * @param x        The input signal.
     * @param nperseg  The length of each segment (window size). Default is 256.
     *                 Pass -1 to use default.
     * @param noverlap The number of points to overlap between segments. Default is
     *                 nperseg // 2. Pass -1 to use default.
     * @param nfft     The FFT size to use. If -1, uses nperseg. Must be >=
     *                 nperseg.
     * @param window   The window function to apply. If null, uses Hann window.
     * @param boundary The boundary extension mode. "zeros" pads with zeros, null
     *                 means no padding.
     * @param padded   Whether to pad the signal on both sides. Default is true.
     * @return A 2D array of complex values [frequency bins][time frames]
     *         representing the STFT.
     */
    public JComplex[][] stft(double[] x, int nperseg, int noverlap, int nfft,
            double[] window, String boundary, boolean padded) {
        // Set defaults
        if (nperseg <= 0)
            nperseg = 256;
        if (noverlap == -1)
            noverlap = nperseg / 2;
        if (nfft <= 0)
            nfft = nperseg;
        if (boundary == null)
            boundary = "zeros";

        // Generate default Hann window if not provided
        if (window == null) {
            window = Windows.hanning(nperseg, false); // hannWindow(nperseg);
        }

        // Calculate window scaling factor to match scipy behavior
        // scipy uses: scale = 1.0 / np.sum(window) for 'spectrum' scaling (default)
        double windowScale = 0.0;
        for (double w : window) {
            windowScale += w;
        }

        // Validate parameters
        if (nfft < nperseg) {
            throw new IllegalArgumentException("nfft must be greater than or equal to nperseg");
        }
        if (noverlap >= nperseg) {
            throw new IllegalArgumentException("noverlap must be less than nperseg");
        }

        double[] signal = x;

        // Apply boundary padding if requested
        if (padded && boundary.equals("zeros")) {
            int padLength = nperseg / 2;
            signal = new double[x.length + 2 * padLength];
            System.arraycopy(x, 0, signal, padLength, x.length);
        }

        int hop = nperseg - noverlap;
        // Calculate number of frames to match scipy behavior
        // scipy includes partial frames at the end
        int numFrames = 0;
        if (signal.length >= nperseg) {
            // This matches scipy's calculation
            numFrames = 1 + (int) Math.ceil((double) (signal.length - nperseg) / hop);
        }

        // Calculate frequency bins (for real signal, only positive frequencies)
        int numFreqBins = nfft / 2 + 1;

        JComplex[][] stftResult = new JComplex[numFreqBins][numFrames];

        // Process each frame
        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            int start = frameIdx * hop;

            // Extract and window the segment
            double[] segment = new double[nfft];
            for (int i = 0; i < nperseg && (start + i) < signal.length; i++) {
                segment[i] = signal[start + i] * window[i];
            }
            // Rest of segment is zero-padded if nfft > nperseg

            // Compute FFT
            JComplex[] fftResult = rfft(segment);

            // Apply scaling after FFT (scipy behavior)
            for (int freqIdx = 0; freqIdx < numFreqBins; freqIdx++) {
                double real = fftResult[freqIdx].getReal() / windowScale;
                double imag = fftResult[freqIdx].getImaginary() / windowScale;
                stftResult[freqIdx][frameIdx] = new JComplex(real, imag);
            }
        }

        return stftResult;
    }

    /**
     * Computes the Inverse Short-Time Fourier Transform (ISTFT).
     * This implementation matches the behavior of scipy.signal.istft with default
     * parameters.
     * 
     * @param stftMatrix  The STFT matrix [frequency bins][time frames].
     * @param nperseg     The length of each segment (window size). Must match the
     *                    STFT parameters.
     * @param noverlap    The number of points to overlap between segments. Must
     *                    match the STFT parameters.
     * @param nfft        The FFT size used. If -1, inferred from stftMatrix
     *                    dimensions.
     * @param window      The window function used in STFT. If null, uses Hann
     *                    window.
     * @param boundary    The boundary extension mode used in STFT. "zeros" means
     *                    padding was used.
     * @param inputLength The expected output length. If -1, inferred from
     *                    stftMatrix.
     * @return The reconstructed time-domain signal.
     */
    public double[] istft(JComplex[][] stftMatrix, int nperseg, int noverlap, int nfft,
            double[] window, String boundary, int inputLength) {
        // Set defaults
        if (nperseg <= 0)
            nperseg = 256;
        if (noverlap == -1)
            noverlap = nperseg / 2;

        int numFreqBins = stftMatrix.length;
        int numFrames = stftMatrix[0].length;

        // Infer nfft from frequency bins
        if (nfft <= 0) {
            nfft = (numFreqBins - 1) * 2;
        }
        if (boundary == null)
            boundary = "zeros";

        // Generate default Hann window if not provided
        if (window == null) {
            window = Windows.hanning(nperseg, false); // hannWindow(nperseg);
        }

        // Calculate window scaling factor to match stft
        // Use sum of window values for 'spectrum' scaling
        double windowScale = 0.0;
        for (double w : window) {
            windowScale += w;
        }

        int hop = nperseg - noverlap;

        // Calculate output length with padding
        int outputLengthWithPad = (numFrames - 1) * hop + nperseg;

        double[] output = new double[outputLengthWithPad];
        double[] windowSum = new double[outputLengthWithPad];

        // Process each frame
        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            int start = frameIdx * hop;

            // Extract frequency frame
            JComplex[] frame = new JComplex[numFreqBins];
            for (int i = 0; i < numFreqBins; i++) {
                frame[i] = stftMatrix[i][frameIdx];
            }

            // Compute inverse FFT
            double[] timeSegment = irfft(frame, nfft);

            // Apply window and overlap-add (no additional scaling needed here)
            for (int i = 0; i < nperseg && (start + i) < outputLengthWithPad; i++) {
                output[start + i] += timeSegment[i] * window[i];
                windowSum[start + i] += window[i] * window[i];
            }
        }

        // Normalize by window sum
        for (int i = 0; i < outputLengthWithPad; i++) {
            if (windowSum[i] > 1e-10) {
                // The factor of windowScale accounts for the scaling applied in STFT
                output[i] = output[i] * windowScale / windowSum[i];
            }
        }

        // Remove padding if it was added
        if (boundary.equals("zeros")) {
            int padLength = nperseg / 2;
            if (outputLengthWithPad > 2 * padLength) {
                double[] unpaddedOutput = new double[outputLengthWithPad - 2 * padLength];
                System.arraycopy(output, padLength, unpaddedOutput, 0, unpaddedOutput.length);
                return unpaddedOutput;
            }
        }

        return output;
    }
}
