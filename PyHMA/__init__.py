import copy
import warnings
import bottleneck as bn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix


class HMA:

    def __init__(self, q_obs, q_sim, b=4, max_lag=3, max_lead=3, measure='nse',
                 keep_internals=False, calc_rays=False):
        """
        Hydrograph Matching Algorithm according to (Ewen 2014). Includes several implementations with different
        requirements in CPU time and RAM. The implementations are given below, in order of preferred use. Typical
        runtime is indicated for 66k observations
        (1) calc_dense2: not suitable if connecting rays between sim and obs are needed (6.5 s).
        (2) calc_dense: allows calculating rays and faster then calc_dense2, but much higher RAM use (4.5 s).
        (3) calc_sparse: if rays are needed and not enough RAM for calc_dense, much slower! (43 s).
        (4) calc_orig: debugging only.
        :param q_obs: np.array or pd.Series, same_type as q_sim
        :param q_sim: np.array or pd.Series, same type as q_obs
        :param b: weighting factor for timing errors
        :param max_lag: maximum lag of simulated values behind observed
        :param max_lead: maximum lead of simulated values ahead of observed
        :param measure: one of ('nse', 'square', 'mae', 'abs'): use HMA based on nash-sutcliffe/squared differences or
        mean absolute error/absolute differences.
        :param keep_internals: boolean, whether to store matrices with work and cumulative work.
        :param calc_rays: also calculate the connecting rays between obs and sim points for plotting.
        """
        if isinstance(q_obs, np.ndarray) and isinstance(q_sim, np.ndarray):
            if len(q_obs) != len(q_sim):
                raise ValueError('(length of) index of obs and sim has to be the same')
            self.q_obs = q_obs
            self.q_sim = q_sim
            self.time = None
        elif isinstance(q_obs, pd.Series) and isinstance(q_sim, pd.Series):
            self.q_obs = q_obs.values
            self.q_sim = q_sim.values
            if not (q_obs.index == q_sim.index).all():
                raise ValueError('(length of) index of obs and sim has to be the same')
            self.time = ((q_sim.index.values - q_sim.index.values[0])/pd.Timedelta('1min'))

        self.b = b
        self.max_lag = max_lag
        self.max_lead = max_lead

        self.measure = measure
        if measure in ('nse', 'square'):
            self.fbench = ((q_obs-q_obs.mean())**2).sum()
        elif measure in ('mae', 'abs'):
            self.fbench = np.abs(q_obs - q_obs.mean()).sum()
        else:
            raise ValueError('HMA measure must be one of nse or mae')
        self.keep_internals = keep_internals

        # Variables for core of HMA
        self.of = np.nan
        self.opt_score = np.nan
        self.aw = None
        self.cw = None
        self.cw0 = None
        self.cw1 = None
        self.v = None

        # Variables for connecting rays
        self.res = np.nan * self.q_obs
        self.tau = np.nan * self.q_obs
        self.rays = [None] * len(self.q_obs)
        self.calc_rays = calc_rays

    def calc_orig(self):
        """
        Execute HMA following precisely the pseudo-code from Ewen (2014). High CPU and RAM usage, so should only be used
        as a reference implementation.
        :return:
        """
        measure = self.measure
        # Calculate work for all possible matching pairs from sim and obs
        aw = np.full(shape=(len(self.q_obs), len(self.q_sim)), fill_value=np.nan)
        # iterate through all flow observations
        for i_o in range(len(self.q_obs)):
            # Check only simulations within the allowed window
            for i_s in range(max(0, i_o - self.max_lead),
                             min(len(self.q_sim), i_o + self.max_lag + 1)):
                if measure in ('nse', 'square'):
                    aw[i_s, i_o] = (self.q_sim[i_s] - self.q_obs[i_o]) ** 2 + self.b ** 2 * (i_s - i_o) ** 2
                elif measure in ('mae', 'abs'):
                    aw[i_s, i_o] = np.abs(self.q_sim[i_s] - self.q_obs[i_o]) + self.b * np.abs(i_s - i_o)

        # Calculate cumulative work along possible paths
        cw = np.ones(shape=aw.shape + (2,)) * np.nan
        # Populate first column
        cw[:, 0, 0] = aw[:, 0]

        if self.keep_internals:
            self.aw = aw
            self.cw = cw

        # Populate other columns
        # Filter out warning to suppress warnings from numpy when np.nanmin
        # encounters all NaN slices
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i_o in range(1, aw.shape[1]):
                for i_s in range(max(0, i_o - self.max_lead),
                                 min(aw.shape[1], i_o + self.max_lag + 1)):
                    # e = aw[i_s, i_o]
                    if measure in ('nse', 'square'):
                        e = (self.q_sim[i_s] - self.q_obs[i_o]) ** 2 + self.b ** 2 * (i_s - i_o) ** 2
                    elif measure in ('mae', 'abs'):
                        e = np.abs(self.q_sim[i_s] - self.q_obs[i_o]) + self.b * np.abs(i_s - i_o)

                    # Repeat the same simulation point
                    cw[i_s, i_o, 1] = e + cw[i_s, i_o - 1, 0]

                    if i_s > 0:
                        m1 = np.nanmin(cw[i_s - 1, i_o - 1, :])
                        if i_s > 1:
                            # Skip a sim point
                            m2 = np.nanmin(cw[i_s - 2, i_o - 1, :])
                        else:
                            m2 = np.inf
                        m = min(m1, m2)
                        cw[i_s, i_o, 0] = e + m
        self.opt_score = np.nanmin(cw[:, -1, :])
        self.of = 1 - self.opt_score / self.fbench

        if self.calc_rays:
            # self._do_calc_rays(cw[:, :, 0], cw[:, :, 1])
            s, e = 0, None
            for i_o in range(len(self.q_obs))[::-1]:
                # print('')
                # print(i_o, s, e)
                # print(cw[s:e, i_o, 0])
                m = np.nanargmin(cw[s:e, i_o, 0]) + s
                try:
                    m_rep = np.nanargmin(cw[s:e, i_o, 1]) + s
                except ValueError:
                    pass
                else:
                    if cw[m_rep, i_o, 1] <= cw[m, i_o, 0]:
                        m = m_rep
                self.tau[i_o] = i_o - m
                self.res[i_o] = self.q_obs[i_o] - self.q_sim[m]
                self.rays[i_o] = [(i_o, self.q_obs[i_o]), (m, self.q_sim[m])]
                # print(i_o, s, e, m)
                s, e = max(0, m - 2), m + 1

        return self.of

    def calc_sparse(self, calc_aw=False, vector_calc=True):
        """
        Partially vectorized implementation, using sparse arrays for cumulative work. Low RAM usage, but relatively high
        CPU time.
        :param calc_aw: if both this and self.keep_internals are true, calculate the matrix aw for work between
        pairs of obs and sim points. Not needed for rest of calculation, debugging only.
        :param vector_calc: use the vectorized version of the calculation. Slower, but code is easier to understand.
        :return:
        """
        # Local copies of instance attributes for faster access
        b = self.b
        qs = self.q_sim
        qo = self.q_obs
        maxlead = self.max_lead
        maxlag = self.max_lag
        measure = self.measure
        time = self.time

        # Prepare sparse matrices with only cells around diagonal defined.
        # padds 1 extra cell on each side to avoid false zeros later
        n = len(self.q_obs)
        w = self.max_lag + self.max_lead + 3
        data = np.array([np.full(n, np.inf)]).repeat(w, axis=0)
        offsets = np.arange(-self.max_lead-1, self.max_lag + 1.5)

        # Calculate work between obs and sim pairs, not needed anymore.
        if calc_aw:
            aw = dia_matrix((data, offsets), shape=(n, n), dtype=np.float64).tocsr()
            st = pd.Timestamp.now()
            # print('Start aw loops')
            # iterate through all flow observations
            for i_o in range(len(qo)):
                # Check only simulations within the allowed window
                for i_s in range(max(0, i_o - self.max_lead),
                                 min(len(qs), i_o + maxlag + 1)):
                    if measure in ('nse', 'square'):
                        aw[i_s, i_o] = (qs[i_s] - qo[i_o]) ** 2 + b ** 2 * (i_s - i_o) ** 2
                    elif measure in ('mae', 'abs'):
                        aw[i_s, i_o] = np.abs(qs[i_s] - qo[i_o]) + b * np.abs(i_s - i_o)
            # print('End aw loops', (pd.Timestamp.now()-st)/pd.Timedelta('1s'))
            self.aw = aw

            # Old debugging code: run original implementation alongside
            # Applies to all variables appended with _orig
            # aw_orig = np.full(shape=(len(self.q_obs), len(self.q_sim)), fill_value=np.nan)
            # # iterate through all flow observations
            # for i_o in range(len(self.q_obs)):
            #     # Check only simulations within the allowed window
            #     for i_s in range(max(0, i_o - self.max_lead),
            #                      min(len(self.q_sim), i_o + self.max_lag + 1)):
            #         aw_orig[i_s, i_o] = (self.q_sim[i_s] - self.q_obs[i_o]) ** 2 + self.b ** 2 * (i_s - i_o) ** 2

        # Calculate cumulative work along possible paths
        cw0 = dia_matrix((data, offsets), shape=(n, n), dtype=np.float64).tocsr()
        cw1 = dia_matrix((data, offsets), shape=(n, n), dtype=np.float64).tocsr()

        # Old debugging code: run original implementation (with dense arrays) alongside
        # Applies to all variables marked _orig
        # # Calculate cumulative work along possible paths
        # cw_orig = np.ones(shape=aw_orig.shape + (2,)) * np.nan
        # # Populate first column
        # cw_orig[:, 0, 0] = aw_orig[:, 0]

        if self.keep_internals:
            # self.aw_orig = aw_orig
            self.cw0 = cw0
            self.cw1 = cw1
            # self.cw_orig = cw_orig

        # Populate other columns
        for i_o in range(n):
            iss = max(0, i_o - maxlead)  # sim index start
            ise = min(n, i_o + maxlag + 1)  # sim index end
            isv = np.arange(iss, ise)  # sim index as vector
            if time is not None:
                dt = time[isv] - time[i_o]
                # Heavily penalize sim-obs combinations outside allowed window
                # so that they are not selected
                dt[(dt < -maxlag) | (dt > maxlead)] = np.inf
            else:
                dt = isv - i_o
            if vector_calc:
                # Vectorized version of the calculation.
                # For understanding it is best to take a look at the else clause below
                # Calculate the work for each (sim, obs_i) pair
                if measure in ('nse', 'square'):
                    e = (qs[iss:ise] - qo[i_o]) ** 2 + b ** 2 * dt ** 2
                elif measure in ('mae', 'abs'):
                    e = np.abs(qs[iss:ise] - qo[i_o]) + b * np.abs(dt)
                if i_o == 0:
                    # Only populate first column of cw0 and move to i_o = 1
                    cw0[iss:ise, 0] = e.reshape(len(e), 1)
                    continue
                # Repeat the same simulation point
                d = cw0[iss:ise, i_o - 1].toarray()
                d[d == 0] = np.nan
                cw1[iss:ise, i_o] = e.reshape(len(e), 1) + d

                # Find the 'cheapest' available preceding sim point
                points = np.full((len(isv), 4), np.inf)
                # Use the previous simulation point
                st = max(0, iss - 1)
                end = min(n + 1, ise - 1)
                l = end - st
                points[-l:, 0] = cw0[st:end, i_o - 1].toarray().squeeze()
                points[-l:, 1] = cw1[st:end, i_o - 1].toarray().squeeze()
                # Skip a simulation point
                st = max(0, iss - 2)
                end = min(n + 1, ise - 2)
                l = end - st
                points[-l:, 2] = cw0[st:end, i_o - 1].toarray().squeeze()
                points[-l:, 3] = cw1[st:end, i_o - 1].toarray().squeeze()
                # points[points == 0] = np.nan
                cw0[iss:ise, i_o] = (e + np.min(points, axis=1)).reshape(len(e), 1)
            else:
                # Old, non-vectorized code. Slower.
                # Left in place since it is easier to understand than vectorized code above
                def zero_to_nan(x):
                    if x == 0:
                        return np.nan
                    else:
                        return x

                for i_s in range(max(0, i_o - self.max_lead),
                                 min(n, i_o + self.max_lag + 1)):
                    # print(i_o, i_s, sep='\t')
                    # e = aw[i_s, i_o]
                    if time:
                        dt = time[isv] - time[i_o]
                        # Heavily penalize sim-obs combinations outside allowed window
                        # so that they are not selected
                        dt[dt > maxlead] = 999999999
                        dt[dt < -maxlag] = 999999999
                    else:
                        dt = isv - i_o
                    if measure in ('nse', 'square'):
                        e = (self.q_sim[i_s] - self.q_obs[i_o]) ** 2 + b ** 2 * dt ** 2
                    elif measure in ('mae', 'abs'):
                        e = np.abs(self.q_sim[i_s] - self.q_obs[i_o]) + np.abs(b * dt)
                    # e_orig = aw_orig[i_s, i_o]
                    # if e != e_orig:
                    #     print('ediff', e, e_orig, sep='\t')

                    # Repeat the same simulation point
                    cw1[i_s, i_o] = e + zero_to_nan(cw0[i_s, i_o - 1])
                    # cw_orig[i_s, i_o, 1] = e + cw_orig[i_s, i_o - 1, 0]
                    # if cw1[i_s, i_o] != cw_orig[i_s, i_o, 1]:
                    #     print('cw1diff', cw1[i_s, i_o], cw_orig[i_s, i_o, 1], sep='\t')

                    if i_s == 0:
                        continue
                    # Else:
                    # Find the 'cheapest' available preceding point
                    # Use the previous simulation point
                    points = [zero_to_nan(cw0[i_s - 1, i_o - 1]),
                              zero_to_nan(cw1[i_s - 1, i_o - 1])]
                    # m1 = np.nanmin(cw_orig[i_s - 1, i_o - 1, :])
                    # m2 = np.inf
                    if i_s > 1:
                        # Skip a simulation point
                        points += [cw0[i_s - 2, i_o - 1], cw1[i_s - 2, i_o - 1]]
                        # m2 = np.nanmin(cw_orig[i_s - 2, i_o - 1, :])
                    # m = min(m1, m2)
                    try:
                        cp = min([p for p in points if p > 0])
                    except ValueError:
                        cp = np.nan
                    # if cp != m:
                    #     print('cpdiff', cp, m, sep='\t')
                    #     print('\torig', cw_orig[i_s - 1, i_o - 1, :], cw_orig[i_s - 2, i_o - 1, :])
                    #     print('\tspar', points)
                    cw0[i_s, i_o] = e + cp
                    # cw_orig[i_s, i_o, 0] = e + m
                    # print('')
        # Find the cheapest point in the last column, i.e. end of cheapest path, i.e. optimum score
        # nz0 = cw0[cw0[:, -1].nonzero()[0], -1].toarray().min()
        # nz1 = cw1[cw1[:, -1].nonzero()[0], -1].toarray().min()
        nz0 = cw0[-self.max_lag:, -1].toarray()
        nz1 = cw1[-self.max_lag:, -1].toarray()
        self.opt_score = min(nz0[nz0 != 0].min(), nz1[nz1 != 0].min())

        self.of = 1 - self.opt_score / self.fbench

        if self.calc_rays:
            self._do_calc_rays(cw0, cw1)

        return self.of

    def calc_dense(self, calc_aw=False):
        """
        Partially vectorized implementation, using dense arrays for cumulative work. High RAM usage, but lowest CPU
        time.
        :param calc_aw: if both this and self.keep_internals are true, calculate the matrix aw for work between
        pairs of obs and sim points. Not needed for rest of calculation, debugging only.
        :return:
        """
        # Local copies of instance attributes for faster access
        b = self.b
        qs = self.q_sim
        qo = self.q_obs
        maxlead = self.max_lead
        maxlag = self.max_lag
        measure = self.measure
        time = self.time
        n = len(self.q_obs)

        # Calculate work between obs and sim pairs, not needed anymore.
        if calc_aw:
            aw = np.full((n, n), np.nan)
            st = pd.Timestamp.now()
            print('Start aw loops')
            # iterate through all flow observations
            for i_o in range(len(qo)):
                # Check only simulations within the allowed window
                for i_s in range(max(0, i_o - self.max_lead),
                                 min(len(qs), i_o + maxlag + 1)):
                    aw[i_s, i_o] = (qs[i_s] - qo[i_o]) ** 2 + b ** 2 * (i_s - i_o) ** 2
            print('End aw loops', (pd.Timestamp.now()-st)/pd.Timedelta('1s'))
            self.aw = aw

            # Old debugging code: run original implementation alongside
            # Applies to all variables appended with _orig
            # aw_orig = np.zeros(shape=(len(self.q_obs), len(self.q_sim))) * np.nan
            # # iterate through all flow observations
            # for i_o in range(len(self.q_obs)):
            #     # Check only simulations within the allowed window
            #     for i_s in range(max(0, i_o - self.max_lead),
            #                      min(len(self.q_sim), i_o + self.max_lag + 1)):
            #         aw_orig[i_s, i_o] = (self.q_sim[i_s] - self.q_obs[i_o]) ** 2 + self.b ** 2 * (i_s - i_o) ** 2

        # Calculate cumulative work along possible paths
        cw0 = np.full((n, n), np.inf, dtype=np.float64)
        cw1 = np.full((n, n), np.inf, dtype=np.float64)

        # Old debugging code: run original implementation (with dense arrays) alongside
        # Applies to all variables marked _orig
        # # Calculate cumulative work along possible paths
        # cw_orig = np.ones(shape=aw_orig.shape + (2,)) * np.nan
        # # Populate first column
        # cw_orig[:, 0, 0] = aw_orig[:, 0]

        if self.keep_internals:
            # self.aw_orig = aw_orig
            self.cw0 = cw0
            self.cw1 = cw1
            # self.cw_orig = cw_orig

        # Filter out warning to suppress warnings from numpy when np.nanmin
        # encounters all NaN slices
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')

        # Populate other columns
        for i_o in range(n):
            iss = max(0, i_o - maxlead)  # sim index start
            ise = min(n, i_o + maxlag + 1)  # sim index end
            isv = np.arange(iss, ise)  # sim index as vector
            if time is not None:
                dt = time[isv] - time[i_o]
                # Heavily penalize sim-obs combinations outside allowed window
                # so that they are not selected
                dt[(dt < -maxlag) | (dt > maxlead)] = np.inf
            else:
                dt = isv - i_o
            # Vectorized version of the calculation.
            # For understanding it is best to take a look at the calc_sparse, where non-vectorized code is kept for
            # reference
            # Calculate the work for each (sim, obs_i) pair
            if measure in ('nse', 'square'):
                e = (qs[iss:ise] - qo[i_o]) ** 2 + b ** 2 * dt ** 2
            elif measure in ('mae', 'abs'):
                e = np.abs(qs[iss:ise] - qo[i_o]) + b * np.abs(dt)
            if i_o == 0:
                # Only populate first column of cw0 and move to i_o = 1
                cw0[iss:ise, 0] = e
                continue

            # Repeat the same simulation point
            d = cw0[iss:ise, i_o - 1]#.toarray()
            d[d == 0] = np.nan
            cw1[iss:ise, i_o] = e + d

            # Find the 'cheapest' available preceding sim point
            points = np.full((len(isv), 4), np.inf)
            # Use the previous simulation point, not repeated
            st = max(0, iss-1)
            end = min(n + 1, ise - 1)
            l = end - st
            points[-l:, 0] = cw0[st:end, i_o - 1]
            # Use the previous simulation point, repeated
            points[-l:, 1] = cw1[st:end, i_o - 1]
            # Skip a simulation point, not repeated
            st = max(0, iss-2)
            end = min(n + 1, ise - 2)
            l = end - st
            points[-l:, 2] = cw0[st:end, i_o - 1]
            # Skip a simulation point, repeated
            points[-l:, 3] = cw1[st:end, i_o - 1]
            # Take best preceding point
            cw0[iss:ise, i_o] = (e + np.min(points, axis=1))
        # Find the cheapest point in the last column, i.e. end of cheapest path, i.e. optimum score
        self.opt_score = min(np.min(cw0[:, -1]), np.min(cw1[:, -1]))
        self.of = 1 - self.opt_score / self.fbench

        if self.calc_rays:
            self._do_calc_rays(cw0, cw1)

        return self.of

    def calc_dense2(self):
        """
        Partially vectorized implementation. Cumulative work array is not stored, only most recent columns
        that are required for the calculation are kept. Faster and uses a lot less memory than calc_dense().
        Note: not possible to calculate connecting rays between sim and obs using this implementation.
        :return:
        """
        # Local copies of instance attributes for faster access
        b = self.b
        qs = self.q_sim
        qo = self.q_obs
        maxlead = self.max_lead
        maxlag = self.max_lag
        measure = self.measure
        time = self.time
        n = len(self.q_obs)

        if self.calc_rays:
            warnings.warn('Not possible to calculate connecting rays when using HMA.calc_dense2')
        if self.keep_internals:
            warnings.warn('Not possible to keep internals when using HMA.calc_dense2')

        for i_o in range(n):
            # Cycle cw arrays
            if i_o > 0:
                cw0_prev = cw0_cur
                cw1_prev = cw1_cur
            cw0_cur = np.full(n, np.inf, dtype=np.float64)
            cw1_cur = np.full(n, np.inf, dtype=np.float64)

            iss = max(0, i_o - maxlead)  # sim index start
            ise = min(n, i_o + maxlag + 1)  # sim index end
            isv = np.arange(iss, ise)  # sim index as vector
            if time is not None:
                dt = time[isv] - time[i_o]
                # Heavily penalize sim-obs combinations outside allowed window
                # so that they are not selected
                dt[(dt < -maxlag) | (dt > maxlead)] = np.inf
            else:
                dt = isv - i_o
            if measure in ('nse', 'square'):
                e = (qs[iss:ise] - qo[i_o]) ** 2 + b ** 2 * dt ** 2
            elif measure in ('mae', 'abs'):
                e = np.abs(qs[iss:ise] - qo[i_o]) + b * np.abs(dt)

            if i_o == 0:
                # Only populate first column of cw0 and move to i_o = 1
                cw0_cur[iss:ise] = e
                continue

            # Repeat the same simulation point
            d = cw0_prev[iss:ise]
            cw1_cur[iss:ise] = e + d

            # Find the 'cheapest' available preceding sim point
            points = np.full((len(isv), 4), np.inf)
            # Use the previous simulation point
            st = max(0, iss-1)
            end = min(n + 1, ise - 1)
            l = end - st
            points[-l:, 0] = cw0_prev[st:end]
            points[-l:, 1] = cw1_prev[st:end]
            # Skip a simulation point
            st = max(0, iss - 2)
            end = min(n + 1, ise - 2)
            l = end - st
            points[-l:, 2] = cw0_prev[st:end]
            points[-l:, 3] = cw1_prev[st:end]
            cw0_cur[iss:ise] = (e + np.min(points, axis=1))

        # Find the cheapest point in the last column, i.e. end of cheapest path, i.e. optimum score
        self.opt_score = min(np.min(cw0_cur), np.min(cw1_cur))
        self.of = 1 - self.opt_score / self.fbench

        return self.of

    def _do_calc_rays(self, cw0, cw1):
        s, e = 0, None
        allow_first = True
        allow_second = True
        for i_o in range(len(self.q_obs))[::-1]:
            ind_first, ind_second = np.inf, np.inf
            cw_first, cw_second = np.inf, np.inf
            if allow_first:
                col_no_rep = cw0[s:e, i_o]
                try:
                    col_no_rep = col_no_rep.toarray()[:, 0]
                except AttributeError:
                    pass
                col_no_rep[col_no_rep == 0] = np.nan
                ind_first = np.nanargmin(col_no_rep)  # position in the allowed window
                cw_first = col_no_rep[ind_first]
                ind_first += s  # absolute position
            if allow_second:
                col_rep = cw1[s:e, i_o]
                try:
                    col_rep = col_rep.toarray()[:, 0]
                except AttributeError:
                    pass
                col_rep[col_rep == 0] = np.nan
                try:
                    ind_second = np.nanargmin(col_rep)  # position in the allowed window
                    cw_second = col_rep[ind_second]
                    ind_second += s  # absolute position
                except ValueError:
                    pass
            if cw_first < cw_second:
                # This is the first visit to this sim points,
                # so for the next i_o a different point should be usd.
                ind = ind_first
                allow_first = True
                allow_second = True
                # Allow skipping one simulation
                s = ind_first - 2
                # But not to repeat same sim point
                e = ind_first
            else:
                # This is the second visit to this sim point, so the next i_o also has
                # to be matched to this sim point.
                ind = ind_second
                allow_first = True
                allow_second = False  # Force to visit first rep of this sim point next iteration
                s = ind_second
                e = ind_second
            e += 1
            s = max(0, s)

            self.tau[i_o] = i_o - ind
            self.res[i_o] = self.q_obs[i_o] - self.q_sim[ind]
            try:
                self.rays[i_o] = [(self.q_obs.index[i_o], self.q_obs[i_o]),
                                  (self.q_obs.index[ind], self.q_sim[ind])]
            except AttributeError:
                self.rays[i_o] = [(i_o, self.q_obs[i_o]),
                                  (ind, self.q_sim[ind])]

    def plot(self, ax=None):
        if not self.calc_rays:
            raise ValueError('Plotting not possible if self.calc_rays is False')
        if ax is None:
            fig, ax = plt.subplots(1)
        handles, labels = [], []
        try:
            h = ax.plot(self.q_obs.index, self.q_obs.values, '-', marker='.', label='obs')
        except AttributeError:
            h = ax.plot(self.q_obs, '-', marker='.', label='obs')
        handles.append(h[0])
        labels.append('obs')
        try:
            h = ax.plot(self.q_obs.index, self.q_sim.values, '-', marker='.', label='sim')
        except AttributeError:
            h = ax.plot(self.q_sim, '-', marker='.', label='sim')
        handles.append(h[0])
        labels.append('sim')
        x = np.hstack([[r[0][0], r[1][0], np.nan] for r in self.rays]).flatten()
        y = np.hstack([[r[0][1], r[1][1], np.nan] for r in self.rays]).flatten()
        h = ax.plot(x, y, color='.4')
        handles.append(h[0])
        labels.append('rays')
        ax.grid()
        ax.legend(handles, labels)
        fig = ax.get_figure()
        return fig, ax
