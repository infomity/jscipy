package com.hissain.jscipy.signal.filter;

/**
 * General linear digital filter, equivalent to {@code scipy.signal.lfilter}.
 * <p>
 * Filters a signal with an IIR or FIR filter defined by numerator coefficients
 * {@code b} and denominator coefficients {@code a}, using the Direct Form II
 * Transposed structure:
 *
 * <pre>
 * a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
 *                       - a[1]*y[n-1] - ... - a[N]*y[n-N]
 * </pre>
 *
 * For an FIR filter (for example, taps designed with
 * {@link FIR#firwin(int, double[], double, boolean)}) pass {@code a = {1.0}}.
 * <p>
 * Unless initial conditions are supplied, the filter starts from rest (all
 * internal states zero), exactly like SciPy.
 */
public final class LFilter {

    private LFilter() {
    }

    /**
     * Result of a filtering operation that also returns the final filter state.
     */
    public static final class Result {
        /** The filtered signal, same length as the input. */
        public final double[] y;
        /**
         * The final filter delay values, of length {@code max(len(a), len(b)) - 1}.
         * Pass it as {@code zi} to continue filtering a following block of samples.
         */
        public final double[] zf;

        Result(double[] y, double[] zf) {
            this.y = y;
            this.zf = zf;
        }
    }

    /**
     * Filters a signal with an IIR or FIR filter, starting from rest.
     * Equivalent to {@code scipy.signal.lfilter(b, a, x)}.
     *
     * @param b The numerator coefficient vector.
     * @param a The denominator coefficient vector. If {@code a[0]} is not 1, both
     *          {@code a} and {@code b} are normalized by {@code a[0]}.
     * @param x The input signal.
     * @return The filtered signal, same length as {@code x}.
     * @throws IllegalArgumentException if {@code b} or {@code a} is null or empty,
     *                                  or {@code a[0]} is zero.
     */
    public static double[] lfilter(double[] b, double[] a, double[] x) {
        return lfilter(b, a, x, null).y;
    }

    /**
     * Filters a signal with an IIR or FIR filter using the given initial
     * conditions, and returns both the output and the final filter state.
     * Equivalent to {@code scipy.signal.lfilter(b, a, x, zi=zi)}.
     *
     * @param b  The numerator coefficient vector.
     * @param a  The denominator coefficient vector. If {@code a[0]} is not 1, both
     *           {@code a} and {@code b} are normalized by {@code a[0]}.
     * @param x  The input signal.
     * @param zi Initial conditions for the filter delays, of length
     *           {@code max(len(a), len(b)) - 1}, or {@code null} for initial rest.
     *           See {@link #lfilter_zi(double[], double[])}.
     * @return A {@link Result} with the filtered signal {@code y} and the final
     *         delay values {@code zf}.
     * @throws IllegalArgumentException if {@code b} or {@code a} is null or empty,
     *                                  {@code a[0]} is zero, or {@code zi} has the
     *                                  wrong length.
     */
    public static Result lfilter(double[] b, double[] a, double[] x, double[] zi) {
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        double[][] ba = normalize(b, a);
        double[] bn = ba[0];
        double[] an = ba[1];
        int n = bn.length;

        double[] z = new double[Math.max(n - 1, 0)];
        if (zi != null) {
            if (zi.length != z.length) {
                throw new IllegalArgumentException(
                        "zi must have length max(len(a), len(b)) - 1 = " + z.length + ", got " + zi.length);
            }
            System.arraycopy(zi, 0, z, 0, z.length);
        }

        double[] y = new double[x.length];
        if (n == 1) {
            // Pure gain, no state.
            for (int i = 0; i < x.length; i++) {
                y[i] = bn[0] * x[i];
            }
            return new Result(y, z);
        }

        // Direct Form II Transposed
        for (int i = 0; i < x.length; i++) {
            double xi = x[i];
            double yi = bn[0] * xi + z[0];
            for (int j = 0; j < n - 2; j++) {
                z[j] = bn[j + 1] * xi + z[j + 1] - an[j + 1] * yi;
            }
            z[n - 2] = bn[n - 1] * xi - an[n - 1] * yi;
            y[i] = yi;
        }
        return new Result(y, z);
    }

    /**
     * Constructs initial conditions for {@code lfilter} for a step response
     * steady state. Equivalent to {@code scipy.signal.lfilter_zi(b, a)}.
     * <p>
     * Multiplying the returned vector by the first sample {@code x[0]} and passing
     * it as {@code zi} makes the filter behave as if the input had been
     * {@code x[0]} forever before the first sample, which reduces start-up
     * transients.
     *
     * @param b The numerator coefficient vector.
     * @param a The denominator coefficient vector (normalized by {@code a[0]} if it
     *          is not 1).
     * @return The initial state {@code zi} of length
     *         {@code max(len(a), len(b)) - 1}.
     * @throws IllegalArgumentException if {@code b} or {@code a} is null or empty,
     *                                  {@code a[0]} is zero, or the filter has no
     *                                  steady state (the sum of {@code a} is zero).
     */
    public static double[] lfilter_zi(double[] b, double[] a) {
        double[][] ba = normalize(b, a);
        double[] bn = ba[0];
        double[] an = ba[1];
        int n = bn.length;
        if (n <= 1) {
            return new double[0];
        }

        double asum = 0.0;
        for (double v : an) {
            asum += v;
        }
        if (asum == 0.0) {
            throw new IllegalArgumentException(
                    "Filter has no steady state: the denominator coefficients sum to zero");
        }

        double[] zi = new double[n - 1];
        double csum = 0.0;
        for (int i = 1; i < n; i++) {
            csum += bn[i] - an[i] * bn[0];
        }
        zi[0] = csum / asum;

        double runA = 1.0;
        double runC = 0.0;
        for (int k = 1; k < n - 1; k++) {
            runA += an[k];
            runC += bn[k] - an[k] * bn[0];
            zi[k] = runA * zi[0] - runC;
        }
        return zi;
    }

    /**
     * Validates the coefficient vectors, normalizes them by {@code a[0]} and pads
     * both to the same length {@code max(len(a), len(b))}.
     *
     * @return {@code {b, a}} as new arrays of equal length.
     */
    private static double[][] normalize(double[] b, double[] a) {
        if (b == null || b.length == 0) {
            throw new IllegalArgumentException("b must be a non-empty array");
        }
        if (a == null || a.length == 0) {
            throw new IllegalArgumentException("a must be a non-empty array");
        }
        if (a[0] == 0.0) {
            throw new IllegalArgumentException("a[0] must be non-zero");
        }
        int n = Math.max(a.length, b.length);
        double a0 = a[0];
        double[] bn = new double[n];
        double[] an = new double[n];
        for (int i = 0; i < b.length; i++) {
            bn[i] = b[i] / a0;
        }
        for (int i = 0; i < a.length; i++) {
            an[i] = a[i] / a0;
        }
        return new double[][] { bn, an };
    }
}
