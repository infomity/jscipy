package com.hissain.jscipy.signal.filter;

import com.hissain.jscipy.Signal;
import com.hissain.jscipy.signal.ConvolutionMode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the public {@code lfilter(b, a, x)} API (scipy.signal.lfilter
 * equivalent), including the FIR use case from discussion #89.
 */
public class LFilterTest {

    private static final double EPS = 1e-12;

    private static double[] testSignal(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = Math.sin(0.1 * i) + 0.5 * Math.cos(0.37 * i) + 0.01 * i;
        }
        return x;
    }

    // ---- FIR (a = {1.0}) ----

    @Test
    public void firMatchesFullConvolutionTruncated() {
        double[] taps = Signal.firwin(31, new double[] { 100.0 }, 1000.0, true);
        double[] x = testSignal(200);

        double[] y = Signal.lfilter(taps, new double[] { 1.0 }, x);

        double[] full = Signal.convolve(x, taps, ConvolutionMode.FULL);
        double[] expected = new double[x.length];
        System.arraycopy(full, 0, expected, 0, x.length);
        assertArrayEquals(expected, y, 1e-10);
    }

    @Test
    public void movingAverage() {
        double[] b = { 1.0 / 3, 1.0 / 3, 1.0 / 3 };
        double[] x = { 3, 6, 9, 12, 15 };
        // starts from rest: [1, 3, 6, 9, 12]
        assertArrayEquals(new double[] { 1, 3, 6, 9, 12 }, LFilter.lfilter(b, new double[] { 1 }, x), EPS);
    }

    @Test
    public void discussion89BandpassExample() {
        // taps = firwin(201, [fc-?, fc+?], pass_zero=False, fs=fs); y = lfilter(taps, 1.0, x)
        double fs = 1000.0;
        double[] taps = Signal.firwin(201, new double[] { 60.0, 140.0 }, fs, false);
        int n = 2000;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            double t = i / fs;
            x[i] = Math.sin(2 * Math.PI * 100 * t) + Math.sin(2 * Math.PI * 300 * t);
        }

        double[] y = Signal.lfilter(taps, new double[] { 1.0 }, x);

        // Linear-phase FIR delay is (numtaps - 1) / 2 = 100 samples; the 300 Hz tone
        // lies in the stopband and must be removed, the 100 Hz tone kept.
        double maxErr = 0;
        for (int i = 400; i < 1800; i++) {
            double expected = Math.sin(2 * Math.PI * 100 * (i - 100) / fs);
            maxErr = Math.max(maxErr, Math.abs(y[i] - expected));
        }
        assertTrue(maxErr < 0.02, "max error vs delayed 100 Hz tone was " + maxErr);
    }

    // ---- IIR ----

    @Test
    public void firstOrderImpulseResponse() {
        double[] x = new double[8];
        x[0] = 1.0;
        double[] y = LFilter.lfilter(new double[] { 1.0 }, new double[] { 1.0, -0.5 }, x);
        for (int i = 0; i < y.length; i++) {
            assertEquals(Math.pow(0.5, i), y[i], EPS);
        }
    }

    @Test
    public void secondOrderKnownValues() {
        // y[n] = x[n] + 0.5*x[n-1] + 0.9*y[n-1] - 0.2*y[n-2]
        double[] b = { 1.0, 0.5 };
        double[] a = { 1.0, -0.9, 0.2 };
        double[] x = { 1, 2, 3, 4 };
        double[] y = LFilter.lfilter(b, a, x);
        double y0 = 1.0;
        double y1 = 2 + 0.5 * 1 + 0.9 * y0;
        double y2 = 3 + 0.5 * 2 + 0.9 * y1 - 0.2 * y0;
        double y3 = 4 + 0.5 * 3 + 0.9 * y2 - 0.2 * y1;
        assertArrayEquals(new double[] { y0, y1, y2, y3 }, y, EPS);
    }

    @Test
    public void normalizesByLeadingDenominatorCoefficient() {
        double[] x = testSignal(50);
        double[] ref = LFilter.lfilter(new double[] { 1.0 }, new double[] { 1.0, -0.5 }, x);
        double[] scaled = LFilter.lfilter(new double[] { 2.0 }, new double[] { 2.0, -1.0 }, x);
        assertArrayEquals(ref, scaled, EPS);
    }

    @Test
    public void denominatorShorterThanNumeratorIsPadded() {
        double[] x = testSignal(40);
        double[] y1 = LFilter.lfilter(new double[] { 0.2, 0.3, 0.1 }, new double[] { 1.0 }, x);
        double[] y2 = LFilter.lfilter(new double[] { 0.2, 0.3, 0.1 }, new double[] { 1.0, 0.0, 0.0 }, x);
        assertArrayEquals(y1, y2, EPS);
    }

    @Test
    public void pureGain() {
        double[] y = LFilter.lfilter(new double[] { 2.0 }, new double[] { 4.0 }, new double[] { 1, 2, 3 });
        assertArrayEquals(new double[] { 0.5, 1.0, 1.5 }, y, EPS);
    }

    @Test
    public void emptyInputGivesEmptyOutput() {
        LFilter.Result r = LFilter.lfilter(new double[] { 1, 1 }, new double[] { 1, -0.5 }, new double[0], null);
        assertEquals(0, r.y.length);
        assertEquals(1, r.zf.length);
    }

    // ---- initial / final conditions ----

    @Test
    public void chainingFinalStateEqualsSingleCall() {
        double[] b = { 0.2, 0.3, 0.1 };
        double[] a = { 1.0, -0.6, 0.2 };
        double[] x = testSignal(120);
        double[] whole = LFilter.lfilter(b, a, x);

        double[] first = new double[70];
        double[] second = new double[50];
        System.arraycopy(x, 0, first, 0, 70);
        System.arraycopy(x, 70, second, 0, 50);
        LFilter.Result r1 = LFilter.lfilter(b, a, first, null);
        LFilter.Result r2 = LFilter.lfilter(b, a, second, r1.zf);

        double[] joined = new double[120];
        System.arraycopy(r1.y, 0, joined, 0, 70);
        System.arraycopy(r2.y, 0, joined, 70, 50);
        assertArrayEquals(whole, joined, EPS);
    }

    @Test
    public void zeroInitialConditionsEqualRest() {
        double[] b = { 0.2, 0.3 };
        double[] a = { 1.0, -0.6 };
        double[] x = testSignal(30);
        assertArrayEquals(LFilter.lfilter(b, a, x), LFilter.lfilter(b, a, x, new double[1]).y, EPS);
    }

    @Test
    public void lfilterZiGivesSteadyStateForConstantInput() {
        double[] b = { 0.2, 0.2 };
        double[] a = { 1.0, -0.6 }; // DC gain = 0.4 / 0.4 = 1
        double[] zi = LFilter.lfilter_zi(b, a);
        double[] x = { 5, 5, 5, 5, 5, 5 };
        double[] zScaled = { zi[0] * 5 };
        double[] y = LFilter.lfilter(b, a, x, zScaled).y;
        for (double v : y) {
            assertEquals(5.0, v, 1e-10);
        }
    }

    @Test
    public void lfilterZiHigherOrderSteadyState() {
        double[] b = { 0.05, 0.1, 0.05 };
        double[] a = { 1.0, -1.2, 0.5 };
        double gain = (0.05 + 0.1 + 0.05) / (1.0 - 1.2 + 0.5);
        double[] zi = LFilter.lfilter_zi(b, a);
        double[] zScaled = { zi[0] * 2, zi[1] * 2 };
        double[] x = { 2, 2, 2, 2, 2, 2, 2, 2 };
        double[] y = LFilter.lfilter(b, a, x, zScaled).y;
        for (double v : y) {
            assertEquals(2.0 * gain, v, 1e-10);
        }
    }

    @Test
    public void lfilterZiMatchesSciPyReferenceValues() {
        // scipy.signal.lfilter_zi([0.2, 0.2], [1, -0.6]) -> [0.8]
        // zi[0] = (b1 - a1*b0) / sum(a) = (0.2 + 0.6*0.2) / 0.4 = 0.8
        double[] zi = LFilter.lfilter_zi(new double[] { 0.2, 0.2 }, new double[] { 1.0, -0.6 });
        assertArrayEquals(new double[] { 0.8 }, zi, EPS);
    }

    @Test
    public void lfilterZiOfPureGainIsEmpty() {
        assertEquals(0, LFilter.lfilter_zi(new double[] { 3.0 }, new double[] { 1.0 }).length);
    }

    // ---- validation ----

    @Test
    public void rejectsInvalidArguments() {
        double[] x = { 1, 2, 3 };
        assertThrows(IllegalArgumentException.class, () -> LFilter.lfilter(null, new double[] { 1 }, x));
        assertThrows(IllegalArgumentException.class, () -> LFilter.lfilter(new double[0], new double[] { 1 }, x));
        assertThrows(IllegalArgumentException.class, () -> LFilter.lfilter(new double[] { 1 }, null, x));
        assertThrows(IllegalArgumentException.class, () -> LFilter.lfilter(new double[] { 1 }, new double[0], x));
        assertThrows(IllegalArgumentException.class, () -> LFilter.lfilter(new double[] { 1 }, new double[] { 0, 1 }, x));
        assertThrows(IllegalArgumentException.class, () -> LFilter.lfilter(new double[] { 1 }, new double[] { 1 }, null));
        assertThrows(IllegalArgumentException.class,
                () -> LFilter.lfilter(new double[] { 1, 1 }, new double[] { 1, -0.5 }, x, new double[3]));
    }

    @Test
    public void lfilterZiRejectsNoSteadyState() {
        // a = [1, -1] sums to zero (integrator)
        assertThrows(IllegalArgumentException.class,
                () -> LFilter.lfilter_zi(new double[] { 1.0, 1.0 }, new double[] { 1.0, -1.0 }));
    }

    // ---- Signal facade ----

    @Test
    public void facadeDelegates() {
        double[] b = { 0.2, 0.3 };
        double[] a = { 1.0, -0.6 };
        double[] x = testSignal(25);
        assertArrayEquals(LFilter.lfilter(b, a, x), Signal.lfilter(b, a, x), EPS);
        assertArrayEquals(LFilter.lfilter_zi(b, a), Signal.lfilter_zi(b, a), EPS);
        LFilter.Result r = Signal.lfilter(b, a, x, Signal.lfilter_zi(b, a));
        assertEquals(x.length, r.y.length);
        assertEquals(1, r.zf.length);
    }

    @Test
    public void existingButterworthLfilterStillWorks() {
        // Overload resolution must keep the original (signal, fs, cutoff, order) API intact.
        double[] x = testSignal(200);
        double[] y = Signal.lfilter(x, 1000.0, 50.0, 4);
        assertEquals(x.length, y.length);
    }
}
