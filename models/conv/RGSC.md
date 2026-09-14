# RGSC diagnosis and experiment plan

Evidence snapshot: 14 September 2026 UTC. The original arm was audited through
its archived iteration 101; the active restart-off arm has separately completed
and been audited through its final iteration 160. This is the maintained diagnosis
and experiment plan requested by the user. Raw measurements remain in run artifacts.
The immediate-win search correction is verified; an RGSC replacement remains
experimental and has not met the evidence gates for adoption.

## Current training arm

Another session switched the run at 03:54:09 UTC on 13 September, as recorded
in the run handoff. Trainer PID 76188 resumed checkpoint 100 with
`control.restart=0`, `search.contempt=0` and capture clock fixed at 200.
The launcher is `runs/conv_g192p10/scripts/launch_arm_off.ps1`.
It completed the configured 161 iterations normally at iteration 160 on
14 September. The trainer exited and released its GPU allocation; its error
log stayed unchanged. No interruption or further training restart was needed.
The original trainer was stopped during iteration 102. Its completed iteration
101 was removed from the active lineage and preserved in `killed_it101/`,
including its checkpoint, window, CSV and
archived original-arm diagnostic (`runs/conv_g192p10/killed_it101/rgsc_diagnostics.json`).
The historical assessment below includes that archived iteration; it must not
be confused with the new arm's iteration 101 or counted in its conversion totals.

CPU comparison of every checkpoint field verified that
`ckpt_000100_clock200.pt` differs from immutable checkpoint 100 only in
`config.rules.clock_min`, from 50 to 200. Model, EMA, optimizer, replay and
control state match. The modified checkpoint contains 205,584,557 bytes, SHA-256
`4834f207b2e7f57af9948202949cb5b9960dac5e3b50b66d8b890548348a1c79`.
Original checkpoint 100 remains unchanged, SHA-256
`9e299d14b4a2253384d3ecfde8559197034c6d34f39ac298ed1c61e13e37648d`.
The rank and regret heads still train. Disabling their use for restarts is
an intervention to evaluate, not evidence that their ranking problem is solved.
Restarts, contempt, clock and actor state changed together, so this arm alone
cannot isolate RGSC's causal contribution. Independent matched tests remain needed.

New-arm iteration 160 completed at 00:21:30 UTC on 14 September. Collection/training took
960.80/250.06 seconds (16.01/4.17 minutes), totaling 20.18 minutes, with
2,052 updates, no skips or rollback and gradient norm 3.08356. Full-search
rows were 48.95833%. Policy/Q/value losses were 1.14971/1.40093/1.39654;
fresh-origin value loss was 1.39654. Restart-origin value loss is logged as
zero because that subgroup is empty, as verified below. Its 205,566,541-byte latest checkpoint
has 290 finite model tensors, 290 finite EMA tensors and 290 finite optimizer
states at step 330,372. Its configuration matches the new arm and learning
rate 0.0000500000 matches the existing cosine schedule. Replay remains at
16 windows and 3,145,728 rows, covering 145–160. Iterations 101–160 have taken
18.67–22.64 minutes. At 114 full-search share increased from 46.35% to 58.85%,
while training time was unchanged. At 115 the share fell to 49.48% and collection
returned to 16.15 minutes; at 116 collection took 15.38 minutes with 46.35% full
search. The timing is consistent with the changing search mix; the slower
collection at 114 did not repeat. At 152 the fastest total in this arm coincides
with its lowest full-search share, while training time remains stable.
The immutable checkpoint 125 contains 205,573,949 bytes, SHA-256
`2b7b1d4b4062fe952d2aa0139009bc8ce7b46382eed10b0d4ffc5138bad4b46c`.
CPU comparison verified every field, including model, EMA, optimizer,
configuration and control, matches the verified latest checkpoint at 125.
The immutable checkpoint 130 contains 205,573,885 bytes, SHA-256
`a6537e1d964b28621b3e16c0e9270b93554afb7e73b0469782f3c1b8a9248f88`.
CPU comparison verified every field matches the verified latest checkpoint
at 130, including model, EMA, optimizer, configuration and control.
The immutable checkpoint 135 contains 205,573,949 bytes, SHA-256
`97eb2fdb76fe7ba0bb50cdb00da9269f0c76d89361d157a28e0b5b7363c7b44e`.
CPU comparison verified every field matches the verified latest checkpoint
at 135, including model, EMA, optimizer, configuration and control.
The immutable checkpoint 140 contains 205,573,885 bytes, SHA-256
`c997bd71ae92e00bab15e03a8ddcd92a8ae94d8f7a8ee9c32c03eb6131a1e8f9`.
CPU comparison verified every field matches the verified latest checkpoint
at 140, including model, EMA, optimizer, configuration and control.
The immutable checkpoint 145 contains 205,573,885 bytes, SHA-256
`2914b0382a2fb3f96629cf3e8bf88c2decf2941594ea0db282e4ad512fb9cf63`.
CPU comparison verified every field matches the verified latest checkpoint
at 145, including model, EMA, optimizer, configuration and control.
The immutable checkpoint 150 contains 205,573,885 bytes, SHA-256
`cd3f9968529fadc34646142eb865f8b4e4c6af7e82ca25bb8d1b169145556286`.
CPU comparison verified every field matches the verified latest checkpoint
at 150, including model, EMA, optimizer, configuration and control.
The immutable checkpoint 155 contains 205,573,821 bytes, SHA-256
`fa1d0d9c93e4e5169ca2460d5769b3b4a7350e0f070c2fea215e067660383a70`.
CPU comparison verified every field matches the verified latest checkpoint
at 155, including model, EMA, optimizer, configuration and control.
The immutable checkpoint 160 contains 205,573,821 bytes, SHA-256
`4c506176f3e596afaa2380bf7f41b696754dc902daf0fe8da7ab58a4724903b0`.
CPU comparison verified every field matches the verified latest checkpoint
at 160, including model, EMA, optimizer, configuration and control.
The immediate-win search source is unchanged. All 192 immediate wins were
selected (101 full, 91 cheap), with all four omission/selection discrepancy
counts zero. The active lineage now has 46,705/46,705 conversions across
25,362,432 rows in 129 corrected iterations: 34,168 through 100 plus this
arm's 12,537 across 101–160, excluding the archived original-arm iteration
101's 417.

The active diagnostic (`runs/conv_g192p10/rgsc_diagnostics.json`) covers
154–160. Its newest iteration matches the log: 526 labelled completed games
and 193,577 rows, with no partial games excluded. There were no maximum-plies
endings at 160. The rolling start at 154 excludes games already underway then,
so older entries 154–157 are censored subsets and do not match their full logs.
In the preceding 136–142 audit, entry 140 passed the pooled
log-match tolerance despite containing 548 games and 198,660 rows, versus its
verified full cohort of 549 games and 199,637 rows. A rounded log match alone
does not prove cohort completeness. Preserve the earlier full-cohort results;
entries 158–160 reproduce their complete cohorts and logs.
At 160 within-game lift is -0.01128, interval [-0.01525, -0.00758];
actor-adjusted lift is -0.00727, interval [-0.01005, -0.00492]. All four
intervals are negative again after the adjusted interval included zero at 159.
Raw lift remains negative for a forty-third consecutive iteration.
Temperature 2 gives -0.00836, interval [-0.01114, -0.00580]; temperature 4
gives -0.00479, interval [-0.00648, -0.00320], its forty-fifth consecutive
negative interval. Raw/adjusted negative-interval totals in this arm are
45/44. The brief adjusted interval crossing at 159 did not establish recovery.
At 159 within-game lift is -0.00616, interval [-0.01023, -0.00170];
actor-adjusted lift is -0.00177, interval [-0.00527, +0.00336]. The adjusted
interval includes zero, ending the 41-iteration streak of all four intervals
being negative at 118–158. Raw lift remains negative for a forty-second
consecutive iteration. Temperature 2 gives -0.00488, interval
[-0.00782, -0.00166]; temperature 4 gives -0.00317, interval
[-0.00484, -0.00132]. Temperature 4 remains negative for a forty-fourth
consecutive iteration. The smaller raw and adjusted point estimates and the
adjusted interval crossing zero do not establish recovery or independent
future utility. Raw/adjusted negative-interval totals in this arm are now
44/43, respectively; the difference is the adjusted interval at 159.
At 158 within-game lift is -0.01066, interval [-0.01468, -0.00710];
actor-adjusted lift is -0.00611, interval [-0.00856, -0.00386]. Both are
negative for a forty-first consecutive iteration. Temperature 2 gives -0.00805,
interval [-0.01082, -0.00563]; temperature 4 gives -0.00443, interval
[-0.00583, -0.00324]. All four pointwise intervals remain negative;
temperature 4 is negative for a forty-third consecutive iteration. This is the
forty-third negative raw/adjusted window in this arm, after 113, 116 and 118–157.
At 157 within-game lift is -0.00915, interval [-0.01282, -0.00554];
actor-adjusted lift is -0.00482, interval [-0.00693, -0.00268]. Both are
negative for a fortieth consecutive iteration. Temperature 2 gives -0.00675,
interval [-0.00910, -0.00436]; temperature 4 gives -0.00373, interval
[-0.00512, -0.00232]. All four pointwise intervals remain negative;
temperature 4 is negative for a forty-second consecutive iteration. This is the
forty-second negative raw/adjusted window in this arm, after 113, 116 and 118–156.
At 156 within-game lift is -0.00970, interval [-0.01358, -0.00609];
actor-adjusted lift is -0.00450, interval [-0.00682, -0.00224]. Both are
negative for a thirty-ninth consecutive iteration. Temperature 2 gives -0.00709,
interval [-0.00980, -0.00460]; temperature 4 gives -0.00460, interval
[-0.00604, -0.00324]. All four pointwise intervals remain negative;
temperature 4 is negative for a forty-first consecutive iteration. This is the
forty-first negative raw/adjusted window in this arm, after 113, 116 and 118–155.
At 155 within-game lift is -0.01100, interval [-0.01558, -0.00711];
actor-adjusted lift is -0.00596, interval [-0.00895, -0.00321]. Both are
negative for a thirty-eighth consecutive iteration. Temperature 2 gives -0.00772,
interval [-0.01099, -0.00480]; temperature 4 gives -0.00450, interval
[-0.00659, -0.00280]. All four pointwise intervals remain negative;
temperature 4 is negative for a fortieth consecutive iteration. This is the
fortieth negative raw/adjusted window in this arm, after 113, 116 and 118–154.
At 154 within-game lift is -0.00718, interval [-0.01044, -0.00361];
actor-adjusted lift is -0.00444, interval [-0.00667, -0.00192]. Both are
negative for a thirty-seventh consecutive iteration. Temperature 2 gives -0.00555,
interval [-0.00819, -0.00292]; temperature 4 gives -0.00352, interval
[-0.00526, -0.00193]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-ninth consecutive iteration. This is the
thirty-ninth negative raw/adjusted window in this arm, after 113, 116 and 118–153.
At 153 within-game lift is -0.01407, interval [-0.01752, -0.01051];
actor-adjusted lift is -0.00931, interval [-0.01183, -0.00691]. Both are
negative for a thirty-sixth consecutive iteration. Temperature 2 gives -0.00966,
interval [-0.01212, -0.00717]; temperature 4 gives -0.00508, interval
[-0.00679, -0.00348]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-eighth consecutive iteration. This is the
thirty-eighth negative raw/adjusted window in this arm, after 113, 116 and 118–152.
At 152 within-game lift is -0.01240, interval [-0.01655, -0.00904];
actor-adjusted lift is -0.00552, interval [-0.00741, -0.00365]. Both are
negative for a thirty-fifth consecutive iteration. Temperature 2 gives -0.00881,
interval [-0.01136, -0.00680]; temperature 4 gives -0.00507, interval
[-0.00620, -0.00403]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-seventh consecutive iteration. This is the
thirty-seventh negative raw/adjusted window in this arm, after 113, 116 and 118–151.
At 151 within-game lift is -0.01269, interval [-0.01610, -0.00927];
actor-adjusted lift is -0.00643, interval [-0.00826, -0.00448]. Both are
negative for a thirty-fourth consecutive iteration. Temperature 2 gives -0.00916,
interval [-0.01141, -0.00700]; temperature 4 gives -0.00521, interval
[-0.00634, -0.00417]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-sixth consecutive iteration. This is the
thirty-sixth negative raw/adjusted window in this arm, after 113, 116 and 118–150.
At 150 within-game lift is -0.00884, interval [-0.01146, -0.00600];
actor-adjusted lift is -0.00567, interval [-0.00735, -0.00389]. Both are
negative for a thirty-third consecutive iteration. Temperature 2 gives -0.00705,
interval [-0.00892, -0.00526]; temperature 4 gives -0.00456, interval
[-0.00577, -0.00347]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-fifth consecutive iteration. This is the
thirty-fifth negative raw/adjusted window in this arm, after 113, 116 and 118–149.
At 149 within-game lift is -0.00961, interval [-0.01363, -0.00485];
actor-adjusted lift is -0.00444, interval [-0.00680, -0.00176]. Both are
negative for a thirty-second consecutive iteration. Temperature 2 gives -0.00657,
interval [-0.00962, -0.00288]; temperature 4 gives -0.00392, interval
[-0.00607, -0.00104]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-fourth consecutive iteration. This is the
thirty-fourth negative raw/adjusted window in this arm, after 113, 116 and 118–148.
At 148 within-game lift is -0.01361, interval [-0.01759, -0.01017];
actor-adjusted lift is -0.00738, interval [-0.00957, -0.00549]. Both are
negative for a thirty-first consecutive iteration. Temperature 2 gives -0.00974,
interval [-0.01294, -0.00715]; temperature 4 gives -0.00556, interval
[-0.00723, -0.00421]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-third consecutive iteration. This is the
thirty-third negative raw/adjusted window in this arm, after 113, 116 and 118–147.
At 147 within-game lift is -0.01176, interval [-0.01480, -0.00918];
actor-adjusted lift is -0.00677, interval [-0.00875, -0.00508]. Both are
negative for a thirtieth consecutive iteration. Temperature 2 gives -0.00856,
interval [-0.01053, -0.00674]; temperature 4 gives -0.00490, interval
[-0.00614, -0.00382]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-second consecutive iteration. This is the
thirty-second negative raw/adjusted window in this arm, after 113, 116 and 118–146.
At 146 within-game lift is -0.01192, interval [-0.01601, -0.00809];
actor-adjusted lift is -0.00673, interval [-0.00916, -0.00441]. Both are
negative for a twenty-ninth consecutive iteration. Temperature 2 gives -0.00866,
interval [-0.01137, -0.00602]; temperature 4 gives -0.00538, interval
[-0.00689, -0.00384]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirty-first consecutive iteration. This is the
thirty-first negative raw/adjusted window in this arm, after 113, 116 and 118–145.
At 145 within-game lift is -0.00676, interval [-0.01026, -0.00318];
actor-adjusted lift is -0.00349, interval [-0.00539, -0.00126]. Both are
negative for a twenty-eighth consecutive iteration. Temperature 2 gives -0.00596,
interval [-0.00876, -0.00354]; temperature 4 gives -0.00432, interval
[-0.00629, -0.00263]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirtieth consecutive iteration. This is the
thirtieth negative raw/adjusted window in this arm, after 113, 116 and 118–144.
At 144 within-game lift is -0.01039, interval [-0.01267, -0.00812];
actor-adjusted lift is -0.00649, interval [-0.00804, -0.00491]. Both are
negative for a twenty-seventh consecutive iteration. Temperature 2 gives -0.00778,
interval [-0.00947, -0.00608]; temperature 4 gives -0.00469, interval
[-0.00570, -0.00364]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-ninth consecutive iteration. This is the
twenty-ninth negative raw/adjusted window in this arm, after 113, 116 and 118–143.
At 143 within-game lift is -0.00733, interval [-0.01013, -0.00430];
actor-adjusted lift is -0.00377, interval [-0.00582, -0.00181]. Both are
negative for a twenty-sixth consecutive iteration. Temperature 2 gives -0.00538,
interval [-0.00723, -0.00341]; temperature 4 gives -0.00389, interval
[-0.00508, -0.00272]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-eighth consecutive iteration. This is the
twenty-eighth negative raw/adjusted window in this arm, after 113, 116 and 118–142.
At 142 within-game lift is -0.00952, interval [-0.01241, -0.00665];
actor-adjusted lift is -0.00482, interval [-0.00664, -0.00298]. Both are
negative for a twenty-fifth consecutive iteration. Temperature 2 gives -0.00759,
interval [-0.00959, -0.00561]; temperature 4 gives -0.00470, interval
[-0.00598, -0.00345]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-seventh consecutive iteration. This is the
twenty-seventh negative raw/adjusted window in this arm, after 113, 116 and 118–141.
At 141 within-game lift is -0.01008, interval [-0.01372, -0.00677];
actor-adjusted lift is -0.00490, interval [-0.00720, -0.00275]. Both are
negative for a twenty-fourth consecutive iteration. Temperature 2 gives -0.00803,
interval [-0.01047, -0.00575]; temperature 4 gives -0.00511, interval
[-0.00665, -0.00377]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-sixth consecutive iteration. This is the
twenty-sixth negative raw/adjusted window in this arm, after 113, 116 and 118–140.
At 140 within-game lift is -0.00743, interval [-0.00992, -0.00497];
actor-adjusted lift is -0.00399, interval [-0.00575, -0.00230]. Both are
negative for a twenty-third consecutive iteration. Temperature 2 gives -0.00569,
interval [-0.00757, -0.00399]; temperature 4 gives -0.00335, interval
[-0.00456, -0.00222]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-fifth consecutive iteration. This is the
twenty-fifth negative raw/adjusted window in this arm, after 113, 116 and 118–139.
At 139 within-game lift is -0.01076, interval [-0.01373, -0.00778];
actor-adjusted lift is -0.00542, interval [-0.00746, -0.00344]. Both are
negative for a twenty-second consecutive iteration. Temperature 2 gives -0.00816,
interval [-0.01069, -0.00585]; temperature 4 gives -0.00564, interval
[-0.00751, -0.00405]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-fourth consecutive iteration. This is the
twenty-fourth negative raw/adjusted window in this arm, after 113, 116 and 118–138.
At 138 within-game lift is -0.01102, interval [-0.01524, -0.00710];
actor-adjusted lift is -0.00749, interval [-0.01108, -0.00431]. Both are
negative for a twenty-first consecutive iteration. Temperature 2 gives -0.00791,
interval [-0.01155, -0.00483]; temperature 4 gives -0.00475, interval
[-0.00757, -0.00244]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-third consecutive iteration. This is the
twenty-third negative raw/adjusted window in this arm, after 113, 116 and 118–137.
At 137 within-game lift is -0.00739, interval [-0.01052, -0.00422];
actor-adjusted lift is -0.00459, interval [-0.00653, -0.00288]. Both are
negative for a twentieth consecutive iteration. Temperature 2 gives -0.00493,
interval [-0.00688, -0.00289]; temperature 4 gives -0.00326, interval
[-0.00454, -0.00203]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-second consecutive iteration. This is the
twenty-second negative raw/adjusted window in this arm, after 113, 116 and 118–136.
At 136 within-game lift is -0.00612, interval [-0.00932, -0.00272];
actor-adjusted lift is -0.00444, interval [-0.00670, -0.00222]. Both are
negative for a nineteenth consecutive iteration. Temperature 2 gives -0.00405,
interval [-0.00614, -0.00186]; temperature 4 gives -0.00237, interval
[-0.00356, -0.00111]. All four pointwise intervals remain negative;
temperature 4 is negative for a twenty-first consecutive iteration. This is the
twenty-first negative raw/adjusted window in this arm, after 113, 116 and 118–135.
At 135 within-game lift is -0.00660, interval [-0.00931, -0.00407];
actor-adjusted lift is -0.00315, interval [-0.00504, -0.00134]. Both are
negative for an eighteenth consecutive iteration. Temperature 2 gives -0.00463,
interval [-0.00641, -0.00287]; temperature 4 gives -0.00302, interval
[-0.00407, -0.00200]. All four pointwise intervals remain negative;
temperature 4 is negative for a twentieth consecutive iteration. This is the
twentieth negative raw/adjusted window in this arm, after 113, 116 and 118–134.
At 134 within-game lift is -0.00761, interval [-0.01088, -0.00450];
actor-adjusted lift is -0.00365, interval [-0.00552, -0.00176]. Both are
negative for a seventeenth consecutive iteration. Temperature 2 gives -0.00604,
interval [-0.00850, -0.00369]; temperature 4 gives -0.00403, interval
[-0.00573, -0.00255]. All four pointwise intervals remain negative;
temperature 4 is negative for a nineteenth consecutive iteration. This is the
nineteenth negative raw/adjusted window in this arm, after 113, 116 and 118–133.
At 133 within-game lift is -0.00874, interval [-0.01227, -0.00588];
actor-adjusted lift is -0.00565, interval [-0.00821, -0.00359]. Both are
negative for a sixteenth consecutive iteration. Temperature 2 gives -0.00631,
interval [-0.00895, -0.00426]; temperature 4 gives -0.00383, interval
[-0.00579, -0.00220]. All four pointwise intervals remain negative;
temperature 4 is negative for an eighteenth consecutive iteration. This is the
eighteenth negative raw/adjusted window in this arm, after 113, 116 and 118–132.
At 132 within-game lift is -0.01073, interval [-0.01562, -0.00681];
actor-adjusted lift is -0.00640, interval [-0.00939, -0.00387]. Both are
negative for a fifteenth consecutive iteration. Temperature 2 gives -0.00781,
interval [-0.01127, -0.00502]; temperature 4 gives -0.00452, interval
[-0.00628, -0.00287]. All four pointwise intervals remain negative;
temperature 4 is negative for a seventeenth consecutive iteration. This is the
seventeenth negative raw/adjusted window in this arm, after 113, 116 and 118–131.
At 131 within-game lift is -0.00869, interval [-0.01152, -0.00623];
actor-adjusted lift is -0.00460, interval [-0.00625, -0.00295]. Both are
negative for a fourteenth consecutive iteration. Temperature 2 gives -0.00639,
interval [-0.00853, -0.00434]; temperature 4 gives -0.00390, interval
[-0.00526, -0.00262]. All four pointwise intervals remain negative;
temperature 4 is negative for a sixteenth consecutive iteration. This is the
sixteenth negative raw/adjusted window in this arm, after 113, 116 and 118–130.
Smaller negative magnitudes than at 130 do not establish durable recovery.
At 130 within-game lift is -0.01114, interval [-0.01576, -0.00748];
actor-adjusted lift is -0.00643, interval [-0.01001, -0.00367]. Both are
negative for a thirteenth consecutive iteration. Temperature 2 gives -0.00763,
interval [-0.01073, -0.00504]; temperature 4 gives -0.00433, interval
[-0.00610, -0.00286]. All four pointwise intervals remain negative;
temperature 4 is negative for a fifteenth consecutive iteration. This is the
fifteenth negative raw/adjusted window in this arm, after 113, 116 and 118–129.
At 129 within-game lift is -0.00570, interval [-0.00892, -0.00219];
actor-adjusted lift is -0.00303, interval [-0.00494, -0.00098]. Both are
negative for a twelfth consecutive iteration. Temperature 2 gives -0.00524,
interval [-0.00759, -0.00300]; temperature 4 gives -0.00341, interval
[-0.00512, -0.00184]. All four pointwise intervals remain negative;
temperature 4 is negative for a fourteenth consecutive iteration. This is the
fourteenth negative raw/adjusted window in this arm, after 113, 116 and 118–128.
Smaller negative magnitudes than at 128 do not establish durable recovery.
At 128 within-game lift is -0.01065, interval [-0.01467, -0.00690];
actor-adjusted lift is -0.00550, interval [-0.00834, -0.00312]. Both are
negative for an eleventh consecutive iteration. Temperature 2 gives -0.00820,
interval [-0.01138, -0.00525]; temperature 4 gives -0.00439, interval
[-0.00597, -0.00289]. All four pointwise intervals remain negative;
temperature 4 is negative for a thirteenth consecutive iteration. This is the
thirteenth negative raw/adjusted window in this arm, after 113, 116 and 118–127.
At 127 within-game lift is -0.00630, interval [-0.00956, -0.00299];
actor-adjusted lift is -0.00441, interval [-0.00681, -0.00204]. Both are
negative for a tenth consecutive iteration. Temperature 2 gives -0.00491,
interval [-0.00699, -0.00275]; temperature 4 gives -0.00244, interval
[-0.00346, -0.00126]. All four pointwise intervals remain negative;
temperature 4 is negative for a twelfth consecutive iteration. This is the
twelfth negative raw/adjusted window in this arm, after 113, 116 and 118–126.
At 126 within-game lift is -0.00511, interval [-0.00766, -0.00227];
actor-adjusted lift is -0.00264, interval [-0.00433, -0.00067]. Both are
negative for a ninth consecutive iteration. Temperature 2 gives -0.00398,
interval [-0.00582, -0.00205]; temperature 4 gives -0.00252, interval
[-0.00343, -0.00160]. All four pointwise intervals remain negative;
temperature 4 is negative for an eleventh consecutive iteration. This is the
eleventh negative raw/adjusted window in this arm, after 113, 116 and 118–125.
The smaller negative magnitudes than at 125 do not establish durable recovery.
At 125 within-game lift is -0.00961, interval [-0.01351, -0.00565];
actor-adjusted lift is -0.00570, interval [-0.00829, -0.00329]. Both are
negative for an eighth consecutive iteration. Temperature 2 gives -0.00696,
interval [-0.00944, -0.00436]; temperature 4 gives -0.00424, interval
[-0.00561, -0.00285]. All four pointwise intervals remain negative;
temperature 4 is negative for a tenth consecutive iteration. This is the
tenth negative raw/adjusted window in this arm, after 113, 116 and 118–124.
At 124 within-game lift is -0.00994, interval [-0.01397, -0.00582];
actor-adjusted lift is -0.00474, interval [-0.00664, -0.00292]. Both are
negative for a seventh consecutive iteration. Temperature 2 gives -0.00702,
interval [-0.01032, -0.00382]; temperature 4 gives -0.00410, interval
[-0.00613, -0.00243]. All four pointwise intervals remain negative;
temperature 4 is negative for a ninth consecutive iteration. This is the
ninth negative raw/adjusted window in this arm, after 113, 116 and 118–123.
At 123 within-game lift is -0.00924, interval [-0.01312, -0.00553];
actor-adjusted lift is -0.00409, interval [-0.00643, -0.00188]. Both are
negative for a sixth consecutive iteration. Temperature 2 gives -0.00693,
interval [-0.00999, -0.00427]; temperature 4 gives -0.00409, interval
[-0.00596, -0.00250]. All four pointwise intervals remain negative;
temperature 4 is negative for an eighth consecutive iteration. This is the
eighth negative raw/adjusted window in this arm, after 113, 116 and 118–122.
At 122 within-game lift is -0.01295, interval [-0.01589, -0.01008];
actor-adjusted lift is -0.00751, interval [-0.00920, -0.00576]. Both are
negative for a fifth consecutive iteration. Temperature 2 gives -0.00835,
interval [-0.01030, -0.00653]; temperature 4 gives -0.00435, interval
[-0.00541, -0.00335]. All four pointwise intervals remain negative;
temperature 4 is negative for a seventh consecutive iteration. This is the
seventh negative raw/adjusted window in this arm, after 113, 116 and 118–121.
At 121 within-game lift is -0.00611, interval [-0.00996, -0.00255];
actor-adjusted lift is -0.00359, interval [-0.00577, -0.00142]. Both are
negative for a fourth consecutive iteration. Temperature 2 gives -0.00509,
interval [-0.00742, -0.00276]; temperature 4 gives -0.00337, interval
[-0.00484, -0.00199]. All four pointwise intervals remain negative;
temperature 4 is negative for a sixth consecutive iteration. This is the
sixth negative raw/adjusted window in this arm, after 113, 116 and 118–120.
At 120 within-game lift is -0.00921, interval [-0.01361, -0.00499];
actor-adjusted lift is -0.00654, interval [-0.00969, -0.00377]. Both are
negative for a third consecutive iteration. Temperature 2 gives -0.00709,
interval [-0.01027, -0.00395]; temperature 4 gives -0.00413, interval
[-0.00603, -0.00228]. All four pointwise intervals remain negative;
temperature 4 is negative for a fifth consecutive iteration. This is the
fifth negative raw/adjusted window in this arm, after 113, 116, 118 and 119.
At 119 within-game lift is -0.00844, interval [-0.01348, -0.00412];
actor-adjusted lift is -0.00394, interval [-0.00687, -0.00129]. Both are
negative for a second consecutive iteration. Temperature 2 gives -0.00616,
interval [-0.01054, -0.00281]; temperature 4 gives -0.00297, interval
[-0.00523, -0.00117]. All four pointwise intervals remain negative;
temperature 4 is negative for a fourth consecutive iteration. This is the
fourth negative raw/adjusted window in this arm, after 113, 116 and 118.
At 118 within-game lift is -0.01166, interval [-0.01544, -0.00777];
actor-adjusted lift is -0.00738, interval [-0.01002, -0.00486]. Both are
negative again. Temperature 2 gives -0.00882, interval [-0.01150, -0.00599];
temperature 4 gives -0.00507, interval [-0.00686, -0.00315]. All four
pointwise intervals are negative; temperature 4 is negative for a third
consecutive iteration. This is the third negative raw/adjusted window in
this arm, following 113 and 116.
At 117 within-game lift is -0.00218, interval [-0.00729, +0.00325];
actor-adjusted lift is -0.00236, interval [-0.00498, +0.00015]. Both include
zero again. Temperature 2 gives -0.00278, interval [-0.00593, +0.00047];
temperature 4 gives -0.00164, interval [-0.00329, -0.000096]. Only the
temperature-4 sensitivity was narrowly negative, for a second iteration.
At 116 within-game lift was -0.00984, interval [-0.01651, -0.00331];
actor-adjusted lift was -0.00592, interval [-0.01118, -0.00131]. Both intervals
were negative again, after including zero at 114–115. Temperature 2 gave
-0.00666, interval [-0.01089, -0.00244]; temperature 4 gave -0.00310,
interval [-0.00524, -0.00068]. All four pointwise intervals were negative.
At 115 raw/adjusted lift was -0.00011/+0.00112 with both intervals spanning
zero; temperature-2/4 intervals also included zero.
At 114 raw/adjusted lift was -0.00038/-0.00000514, with both intervals spanning
zero; its temperature-2/4 intervals also included zero. Those results ended
the preceding negative-interval streaks without establishing recovery.
At 113 within-game lift was -0.00919, interval [-0.01619, -0.00341];
actor-adjusted lift was -0.00415, interval [-0.00906, -0.00032]. These were
the first negative raw and actor-adjusted intervals at configured temperature 1
in this arm. Temperature 2 gave -0.00872, interval [-0.01366, -0.00479];
temperature 4 gave -0.00520, interval [-0.00790, -0.00306]. This was the
second consecutive negative temperature-2 interval and third negative
temperature-4 interval. Temperature 2/4 intervals were positive at 102–106
and included zero at 107–110. At 111, temperature 2 included zero while
temperature 4 was negative; both were negative at 112. These are played-position
diagnostics against the current target, not independent future-utility or strength tests.
Previously verified raw/adjusted lifts were +0.02838/+0.01094 at 102 and
+0.01805/+0.01316 at 103, +0.02496/+0.01453 at 104 and +0.02502/+0.01524
at 105, +0.01825/+0.01031 at 106 and +0.00737/+0.00417 at 107, with positive
intervals; both intervals at 101 included zero. At 108 the raw/adjusted lifts
were +0.00377/+0.00313, with both intervals including zero, ending the
six-window positive-interval streak. At 109 raw/adjusted lift was
+0.01290/+0.00650, and at 110 +0.00715/+0.00337, with positive intervals.
At 111 raw/adjusted lift was +0.00361/+0.00262, and at 112 -0.00078/+0.00037,
all with intervals including zero. The negative results at 113 did not repeat
at 114–115, negativity returned at 116, configured-temperature intervals
included zero at 117 and were negative again at 118–158. At 159 the adjusted
interval includes zero while the raw interval remains negative; at 160 both
intervals are negative again. These results
supply no durable recovery evidence and do not independently establish the cause of the changes.
The rebound at 109–110 has not established a durable RGSC repair or causal
strength benefit from disabling restarts.
At 160 pooled lift is -0.32843, interval [-0.33287, +0.01974]. Row ESS is
4.61737 and game ESS is 1.00689, with maximum row mass 31.95% and maximum
game mass 99.65703%. One game dominates and the interval includes zero;
these measurements do not establish independent future utility or recovery.
At 159 pooled lift is -0.21945, interval [-0.36036, -0.07233]. Row ESS is
3.13144 and game ESS is 2.27444, with maximum row mass 52.12% and maximum
game mass 58.80%. Concentration remains high despite the pooled interval
being negative. The preceding positive pooled score did not persist; neither
sign independently validates future utility or establishes strength.
At 158 pooled lift is +0.02688, interval [-0.16298, +0.03037]. Row ESS is
3.31238 and game ESS is 1.00000000012, with maximum row mass 42.77% and
maximum game mass rounding to 100%. One game dominates and the interval
includes zero. The positive point estimate does not establish recovery or
independent future utility; all four within-game intervals remain negative.
At 157 pooled lift is -0.27182, interval [-0.30131, +0.01936]. Row ESS is
3.63195 and game ESS is 1.32274, with maximum row mass 45.44% and maximum
game mass 86.57%. Concentration remains high and the interval includes zero;
these measurements do not establish independent future utility or recovery.
At 156 pooled lift is -0.22456, interval [-0.22852, +0.01638]. Row ESS is
1.13232 and game ESS is 1.00317, with maximum row mass 93.82% and maximum
game mass 99.84%. One game dominates and the interval includes zero; these
measurements do not establish independent future utility or recovery.
At 155 pooled lift is -0.10063, interval [-0.17071, +0.11571]. Row ESS is
1.89080 and game ESS is 1.58107, with maximum row mass 70.72% and maximum
game mass 75.84%. Concentration remains high and the interval includes zero.
The preceding positive jump did not persist; neither sign establishes
independent future utility or recovery.
At 154 pooled lift is +0.37182, interval [-0.06413, +0.54144]. Row ESS is
2.51878 and game ESS is 1.85022, with maximum row mass 61.28% and maximum
game mass 64.25%. The positive point estimate remains highly concentrated
and its interval includes zero. It does not establish recovery or independent
future utility; all four within-game intervals remain negative.
At 153 pooled lift is -0.06641, interval [-0.18518, +0.01933]. Row ESS is
12.70210 and game ESS is 1.98103, with maximum row mass 16.24% and maximum
game mass 58.05%. Row concentration is lower than in the preceding iterations,
but game concentration remains high and the interval includes zero. These
measurements do not establish independent future utility or recovery; all four
within-game intervals remain negative.
At 152 pooled lift was -0.01968, interval [-0.13770, +0.01065]. Row ESS was
2.75676 and game ESS is 1.75592, with maximum row mass 56.37% and maximum
game mass 73.98%. The estimate remains highly concentrated and its interval
includes zero. Its smaller magnitude does not establish independent future
utility or recovery; all four within-game intervals remain negative.
At 151 pooled lift was -0.27452, interval [-0.27740, +0.03023]. Row ESS was
2.83265 and game ESS is 1.00025, with maximum row mass 45.75% and maximum
game mass 99.98742%. One game dominates and the interval includes zero.
This does not establish independent future utility or recovery; all four
within-game intervals remain negative.
At 150 pooled lift was -0.32211, interval [-0.32526, +0.03363]. Row ESS was
2.06068 and game ESS is 1.00011, with maximum row mass 59.51% and maximum
game mass 99.99452%. One game dominates and the interval includes zero.
The changing pooled sign does not establish recovery or independent future
utility; all four within-game intervals remain negative.
At 149 pooled lift was +0.09374, interval [-0.01452, +1.27879]. Row ESS was
2.03846 and game ESS is 1.00035, with maximum row mass 52.51% and maximum
game mass 99.98252%. One game dominates; its positive point estimate and broad
interval do not establish recovery. All four within-game intervals remain negative.
At 148 pooled lift was +0.02996, interval [-0.17734, +0.03745]. Row ESS was
1.00312 and game ESS rounds to exactly 1.0, with maximum row mass 99.84445%
and maximum game mass rounding to 100%. Almost all weight comes from one
position. The positive point estimate does not establish recovery: its interval
includes zero and all four within-game intervals remain negative.
At 147 pooled lift was -0.02484, interval [-0.18087, +0.03174]. Row ESS was
2.32305 and game ESS is 2.23761, with maximum row mass 63.93% and maximum
game mass 64.11%. The estimate remains highly concentrated and its interval
includes zero. Its smaller magnitude does not establish recovery or independent
future utility; all four within-game intervals remain negative.
At 146 pooled lift was -0.37125, interval [-0.37591, +0.02780]. Row ESS was
1.83807 and game ESS is 1.00000767, with maximum row mass 70.31% and maximum
game mass 99.999617%. One game dominates and the interval includes zero;
the smaller pooled magnitude does not establish recovery or independent
future utility. All four within-game intervals remain negative.
At 145 pooled lift was -0.42515, interval [-0.42895, +0.07080]. Row ESS was
1.90657 and game ESS is 1.00552, with maximum row mass 71.26% and maximum
game mass 99.7249%. One game dominates and the interval includes zero;
the changing pooled sign and smaller within-game negative magnitudes do not
establish recovery. All four within-game intervals remain negative.
At 144 pooled lift was +0.09414, interval [-0.16074, +0.14689]. Row ESS was
1.64540 and game ESS is 1.62588, with maximum row mass 77.50% and maximum
game mass 77.54%. The positive point estimate remains highly concentrated,
its interval includes zero, and all four within-game intervals are negative.
It does not establish recovery.
At 143 pooled lift was +0.00971, interval [-0.12361, +0.19338]. Row ESS was
1.00000767 and game ESS is 1.00000677, with maximum row mass 99.999616%
and maximum game mass 99.999662%. Almost all weight comes from one position;
the positive point estimate and smaller within-game negative magnitudes do not
establish recovery. All four within-game intervals remain negative.
At 142 pooled lift was -0.01012, interval [-0.20600, +0.02667]. Row ESS was
1.79500 and game ESS is 1.66809, with maximum row mass 72.46% and maximum
game mass 72.48%. Its interval includes zero and concentration remains extreme;
the smaller pooled magnitude does not establish recovery.
At 141 pooled lift was -0.15746, interval [-0.19538, +0.00993]. Row ESS was
4.00703 and game ESS is 1.58189, with maximum row mass 41.44% and maximum
game mass 78.60%. Its interval includes zero and concentration remains extreme;
the point estimate does not independently establish future utility or strength.
At 140 pooled lift was -0.08167, interval [-0.10028, +0.03116]. Row ESS was
2.37576 and game ESS is 1.47614, with maximum row mass 61.42% and maximum
game mass 81.62%. Its interval includes zero and concentration remains extreme;
the smaller negative within-game magnitudes do not establish durable recovery.
At 139 pooled lift was +0.08890, interval [-0.25210, +0.10088]. Row ESS was
6.28318 and game ESS is effectively 1.0, with maximum row mass 23.99% and
maximum game mass rounding to 100%. The positive point estimate remains
dominated by one game, its interval includes zero, and all four within-game
intervals remain negative. It does not establish recovery.
At 138 pooled lift was -0.18427, interval [-0.18835, +0.08839]. Row ESS was
2.18918 and game ESS 1.00000001, with maximum row mass 57.40% and maximum
game mass 99.9999995%. Its interval includes zero and one game dominates;
the changing pooled sign and magnitude do not establish recovery.
At 137 pooled lift was -0.01455, interval [-0.29471, +0.02069]. Row ESS was
1.80039 and game ESS 1.26010, with maximum row mass 72.75% and maximum
game mass 88.40%. Its interval includes zero and concentration remains extreme;
the smaller pooled magnitude does not establish recovery.
At 136 pooled lift was -0.16706, interval [-0.17707, +0.03394]. Row ESS was
1.17308 and game ESS 1.13028, with maximum row mass 92.17% and maximum
game mass 93.91%. Its interval includes zero and concentration remains extreme.
At 135 pooled lift was +0.06524, interval [-0.22732, +0.06904]. Row ESS was
1.21522 and game ESS 1.00000252, with maximum row mass 90.53% and maximum
game mass 99.999874%. Its positive point estimate remains dominated by one
game, its interval includes zero, and all four within-game intervals remain
negative. It does not establish recovery.
At 134 pooled lift was -0.38230, interval [-0.38687, -0.01678]. Row ESS was
1.00008 and game ESS 1.00000003, with maximum row mass 99.99590% and maximum
game mass 99.9999985%. Its negative interval remains dominated by one game;
the sign does not remove this concentration limitation.
At 133 pooled lift was +0.61256, interval [-0.11456, +0.61590]. Row ESS was
1.65704 and game ESS rounds to exactly 1.0, with maximum row mass 74.59%
and maximum game mass rounding to 100%. The positive pooled jump is dominated
by one game, its interval includes zero, and all four within-game intervals
remain negative. It does not establish recovery.
At 132 pooled lift was -0.21635, interval [-0.22339, +0.04857]. Row ESS was
2.94369 and game ESS 1.03939, with maximum row mass 40.40% and maximum
game mass 98.07%. Its interval includes zero and game concentration remains
extreme. At 131 pooled lift was -0.08252, interval [-0.30040, -0.07537]. Row ESS was
1.64870 and game ESS 1.00000020, with maximum row mass 73.09% and maximum
game mass 99.999990%. Its negative interval does not remove the extreme
concentration limitation or independently validate future utility.
At 130 pooled lift was -0.17776, interval [-0.38127, +0.03622]. Row ESS was
5.55 and game ESS 1.09544, with maximum row mass 31.63% and maximum game
mass 95.48%. Game concentration remains extreme, and the pooled interval
includes zero. At 129 pooled lift was +0.04026, interval [-0.10645, +0.20500],
with game ESS 1.95168 and maximum game mass 59.51%.
At 128 pooled lift was +0.08218, interval
[-0.16808, +0.08623], with game ESS 1.00001 and maximum game mass 99.99951%.
At 127 pooled lift was -0.12307, interval [-0.15161, +0.02464], with
game ESS 1.36493 and maximum game mass 84.32%.
At 126 pooled lift was +0.01495, interval [-0.23516, +0.02748], with
game ESS 1.08586 and maximum game mass 95.89%. The jump to +0.57385 at 125,
with game ESS 1.51 and an interval including zero, did not persist. These
pooled fluctuations do not establish recovery while the raw and actor-adjusted
within-game intervals remain negative.
At 119 pooled lift was +0.05351 with an interval including zero and game ESS
effectively 1.00; its positive point estimate did not establish recovery.
At 113 pooled lift was -0.19376 with a negative interval, game ESS 1.00006
and maximum game mass 99.997%; that negative sign did not remove its
concentration limitation. The maximum rank logit among newly completed games
at 160 is 20.625 and the median is -21.375. These are scores
from games spanning multiple actor windows, not solely current-head outputs;
changes in their maxima do not establish current-head divergence or recovery.
The current rolling target audit found zero label or perspective mismatches
in 945,910 complete-game rows across 154–160, including 193,577 at 160.
Its total reflects the shifted reconstruction boundary. The prior
101–107 audit checked 1,166,553 rows without mismatches; overlapping audits
must not be summed as independent or unique observations.

Astra independently reconstructed the 120 cohort after the third consecutive
negative raw/adjusted interval. A separate parent CPU calculation verified
543 games, 198,999 rows, zero partial exclusions, both means, sign counts,
medians and deletion/trimmed means. Raw lifts are negative in 386/543 games
(71.1%) and adjusted lifts in 370/543 (68.1%); their medians are -0.00032629
and -0.00010412. The observed deficit is broader and less sensitive to a few
extreme games than the earlier 113 result.

| Iteration-120 sensitivity | Raw lift [95% interval] | Actor-adjusted lift [95% interval] |
|---|---|---|
| Original | -0.009207 [-0.013605, -0.004993] | -0.006541 [-0.009691, -0.003774] |
| Remove worst 1 | -0.008265 [-0.011989, -0.004332] | -0.005602 [-0.007903, -0.003207] |
| Remove worst 5 | -0.006178 [-0.009561, -0.002710] | -0.004186 [-0.005982, -0.002194] |
| Remove worst 10 | -0.005052 [-0.008210, -0.001849] | -0.003467 [-0.005212, -0.001602] |
| Symmetric 10% trim | -0.006597 [-0.008668, -0.004349] | -0.003669 [-0.004911, -0.002388] |

Each deletion orders games by its own metric. Trimming removes 54 games per
tail and is recomputed inside each of 1,000 game-bootstrap replicates, seed 120.
Both intervals remain negative after deleting the worst ten games or trimming.
Actor-119 and actor-120 normalized within-segment lifts are -0.00797086 and
-0.00831324, using segment row fractions within each game and equal game weights.
The 220 win-ending games have raw/adjusted means -0.02013677/-0.01492469;
the 323 clock-ending games have -0.00176304/-0.00083081. Actor mixing and
one ending type therefore do not explain away the observed negative sign.
The review found no new reconstruction or normalization issue. This is a
post-hoc sensitivity check on the same saved played-position cohort, with
repeated looks and shared actors/openings. It strengthens the evidence of
weak ranking against the current target, without independently validating
future utility, proving the cause or establishing a strength effect.

Astra independently reconstructed the 113 cohort; a separate parent CPU check
matched both means, sign counts and deletion means. Raw lifts are negative in
348/545 games and adjusted lifts in 325/545, with medians -0.00012435 and
-0.00003752. The worst game, environment 274, has 260 plies and a decisive
ending, spans actors 112/113 and has raw/adjusted lifts -1.460724/-0.951922.
Thus the magnitude is amplified by a tail, while small negative lifts are broader.

Post-hoc removal of the worst raw game leaves -0.006523, interval
[-0.010888, -0.002251]; removing five leaves -0.004019, interval
[-0.007270, -0.000769]; removing ten leaves -0.002431, interval
[-0.005620, +0.000740]. Removing the worst adjusted game leaves -0.002409,
interval [-0.005629, +0.000378], so adjusted untrimmed significance is
one-game-sensitive. Removing five/ten adjusted games gives -0.000455/+0.000634,
both with intervals spanning zero. Each deletion orders games by its own metric.
Symmetric 10% trimming, removing 54 games per tail and recomputing the trim
inside each bootstrap draw, gives raw -0.002163 [-0.003904, -0.000449] and
adjusted -0.001026 [-0.001959, -0.000074]. These are sensitivity estimates,
not independently confirmed effects or replacements for the original means.

Actor-112 and actor-113 segments contribute -0.002527/-0.001524 to the
adjusted cohort mean; their normalized within-segment lifts are -0.005223/-0.004825.
Weights are segment row fractions within each game, followed by equal game
weights, not pooled row weights. Both decisive and clock-ending subgroup means
are negative. Actor mixing therefore does not fully explain the observed sign.
The review uses 1,000 game-bootstrap replicates with seed 113. Repeated looks,
post-hoc sensitivities and shared actors/openings limit these pointwise intervals;
they do not establish an independent temporal trend or future-utility benefit.
Separate inspection of newly collected window 113 found rank minimum/median/maximum
-46.75/-13.5/24.625, distinct from its completed-game cohort's -56.0/-13.25/24.625.

Astra independently reconstructed all 576 games and 203,134 rows at 108.
That iteration's pooled +0.506965, interval [-0.350507, +0.514745], had
row/game ESS 1.0182/1.0182 and maximum row/game masses 99.0991%/99.0991%.
It decomposes into -0.070576 from weighted within-game
selection and +0.577540 from between-game label means. The dominant game
has 99.0991% of selection mass, mean label +0.574991 and rank-weighted
label +0.503922: its own within-game lift is -0.071069, or -0.138414 after
actor adjustment. Thus a favorable game mean drives the pooled jump even
though selection within that game is worse than uniform.

Deleting the highest-mass game gives pooled lift -0.108420, interval
[-0.355794, +0.020278], game ESS 1.744. Deleting two gives -0.341515,
interval [-0.357997, +0.022881], game ESS 1.068; deleting five gives
-0.153975, interval [-0.175899, +0.008122], game ESS 1.425. Of 1,000 original
bootstrap draws, 646 include the dominant game and are all positive. Of the
354 without it, 89.5% are negative, interval [-0.355874, +0.020264]. These
post-hoc deletion checks expose fragility; they do not establish negative
population ranking, and selection remains concentrated through five removals.
Fixed larger deletions broaden the remainder: removing ten gives +0.009378,
interval [+0.002825, +0.015521], game ESS 52.74 and maximum game mass 5.001%;
removing twenty gives +0.010596, interval [+0.004217, +0.016842], game ESS
75.59 and maximum game mass 3.635%. The broader remainder is mildly positive,
but these post-hoc exclusions do not establish out-of-sample performance.

The dominant 32.5 score is from actor 107, window 107 row 174827 (step 170,
environment 747), ply 289, Q -0.549316, label +0.503922 and action 31.
Its 357-ply game spans 106–108 and ends at window 108 row 46827 (step 45,
environment 747), ply 356, by the capture clock with terminal return -0.05
from the last mover's perspective. Separate CPU inspection verified that row and
the actual collected window 108 score range: minimum -52.0, median
-12.875, maximum 27.375 at row 95072 (step 92, environment 864).
These observations reinforce the distinction between completion-cohort
extrema and scores produced by the newest actor.

For context, pooled lift at 105 was +0.04137, interval [+0.01439, +0.08023],
with row ESS 1.43, game ESS effectively 1.00, maximum row mass 81.74% and
essentially 100% of pooled mass in one game. Its completion-cohort maximum
rank logit was 97.50 and median -10.94. That result prompted the following
separate sensitivity investigation.

Astra's additional CPU reconstruction and sensitivity review reproduced the
105 pooled statistic and all 1,000 original bootstrap draws. Of those draws,
625 included the dominant game, interval [+0.036203, +0.046148]; 375 omitted
it, interval [+0.012373, +0.081475]. All were positive. Whole-game resampling
renormalizes the observed weights correctly; the positive pooled interval
is not itself a bootstrap error. Deleting the largest-mass game gives lift
+0.077139, interval [+0.011814, +0.081759], but game ESS remains 1.00.
Deleting the two largest gives +0.016551, interval [+0.011130, +0.021799],
game ESS 87.33 and maximum game mass 2.98%. Deleting five gives +0.016706,
interval [+0.011478, +0.022654], game ESS 92.05. Positive observed pooled
lift therefore survives removal of these extremes. These are post-hoc
sensitivity checks on saved games, not independent confirmation.

The original pooled lift decomposes as +0.009196 from weighted within-game
selection plus +0.032171 from differences between game means: approximately
78% of the pooled lift is between games. The second extreme game has
within-game lift -0.157102; its favorable game-level mean makes its pooled
comparison positive. Pooled and within-game metrics answer different
selection questions, so neither should substitute for the other.

The 97.5 score was collected during window 103 at row 113084 (step 110,
environment 444), ply 332, capture age 6/200, full search, Q -0.175523,
suffix label +0.025220 and selected action 10. This 685-ply game spans
101–105 and ends at 105 step 78 by the capture clock, with terminal return
+0.05. Its actor-adjusted within-game lift is +0.002114. Actual newly
collected window 105 has maximum score 32.75 and median -12.125; 72.54%
of the completion cohort's 190,402 rows came from older actor windows.
The completion-cohort maxima therefore do not establish temporal divergence
of successive current heads. Delayed outcomes and mixed actor scores explain
part of the apparent instability. At that review, observed within-game ranking
had recovered over four windows, while pooled concentration remained extreme. The current
head remains a serious baseline for the planned target-by-objective experiment;
this review does not establish causal strength benefit or resolve target semantics.

Restarts and newly collected replayed rows are both zero. Logged restart
keys, lost feedback and maximum restart reuse are zero. There are no restart
tree observations, so the zero tree-error field supplies no calibration
evidence. The comparison at 151 used immutable checkpoint 150: 512 entries
with 508/3/1 having zero/one/multiple observations, no scored zero-priority
entries, no restart keys or games, no lost keys or games and no first-tree
observations. The zero lost-key count matches the log.
The historical comparison at 146 used immutable checkpoint 145: 512 entries
with 508/3/1 having zero/one/multiple observations, no scored zero-priority
entries, no restart keys or games, no lost keys or games and no first-tree
observations. The zero lost-key count matches the log.
The historical comparison at 141 used immutable checkpoint 140: 512 entries
with 507/3/2 having zero/one/multiple observations, no scored zero-priority
entries, no restart keys or games, no lost keys or games and no first-tree
observations. The zero lost-key count matches the log.
The historical comparison at 136 used immutable checkpoint 135: 512 entries
with 506/4/2 having zero/one/multiple observations, no scored zero-priority
entries, no restart keys or games, no lost keys or games and no first-tree
observations. The zero lost-key count matches the log.
The historical comparison at 131 used immutable checkpoint 130:
512 entries with 507/3/2 having zero/one/multiple observations, no scored
zero-priority entries, no restart keys or games, no lost keys or games and
no first-tree observations. The zero lost-key count matches the log.
The comparison at 156 used immutable checkpoint 155: 512 entries with
510/2/0 having zero/one/multiple observations, no scored zero-priority entries,
no restart keys or games, no lost keys or games and no first-tree observations.
The zero lost-key count matches the log; it supplies no restart calibration
feedback in this arm.
At 160 buffer scored/tree fractions are 0.00%/99.61%, with 512 boards;
reported draw ESS is 419.80, top-16 mass 8.32%, clock remaining 174.38 and
dominant clock-limit share 100.00%. No entry has multiple opening
observations. Maximum retained priority is 2.51562. With restarts disabled
these retained-buffer summaries are not measurements of positions actually
used for restarting.

Independent Astra and parent CPU comparisons of immutable 150 to latest 152
found all 476 retained entries identical in every saved field. Astra also
matched independently constructed keys to `State.key()` for all entries,
with no duplicate keys. Thirty-six entries left: 34 with no outcome
observations, one with a single observation and the last scored entry. All
36 new entries are unobserved tree candidates with clock limit 200. Observation
counts therefore changed from 508/3/1 to 510/2/0. The removed scored entry
had 31 observations, clock limit 186, capture age 112 and ply 200; every
current entry has clock limit 200. The final fall in scored fraction and rise
in modal clock share reflect buffer turnover with feedback disabled, not
count corruption or improved head calibration. This is an endpoint comparison
across two iterations, not an inventory of every intermediate insertion.

Source review and independent CPU buffer comparisons explain the falling
scored fraction without observation-count corruption. In
[control.py](py/conv/control.py), scored means at least two opening outcomes;
a new tree candidate enters with a predicted priority and zero observations,
while a played candidate brings one. Fresh-game candidate insertion continues
with restarts disabled, but the usual repeated-opening feedback does not.
Between 110 and 111, all 472 retained entries are unchanged in every saved
field. Forty entries were replaced: 29 scored and 11 unscored entries left;
40 zero-observation tree entries, all clock 200, entered. This exactly accounts
for scored entries falling from 351 to 322. At 111 observation counts were
184/6/322 for zero/one/two-or-more outcomes. Across 105–111, all 116 lost
scored entries were evicted; surviving observation counts did not change.
Exact duplicate played candidates could add observations, but none did so
for retained entries in these comparisons.

Unscored priorities average 0.11976 versus 0.03539 for scored entries at 111.
Their provisional predictions can displace lower empirical priorities under
the existing admission rule; this supplies no evidence of calibration.
The tree flag records insertion origin, not whether an entry remains unscored.
`buffer_dropped=0` counts explicit removal of truncated restart openings and
does not count capacity evictions. `buffer_clock_max_frac` is the most common
clock-limit category's share, not the fraction with a full remaining clock.
The modal category is 200: 155/512 entries at 110 and 186/512 at 111; retained
limits are unchanged. At 111, mean remaining clock was 140.01 and mean clock
limit was 162.38. These trends reflect dormant-buffer turnover under this arm; this
review found no separate turnover fault requiring a live training change.

CPU reconstruction with the existing `Replay.restore(116)` loads exactly
windows 101–116 and 3,145,728 rows, with zero `replayed` rows. Source inspection
confirms the current window is added and the oldest trimmed before training.
Original-arm window 100, containing 68,854 restarted-game rows, has aged out
of sampling and remains safely on disk. `value_fresh` and `value_replay` split
training losses by game origin, not by window age: initial-position games versus
games restarted by search control. The empty latter subgroup logs zero through
the masked loss and count-clamped reporting; this is neither perfect prediction
nor absent replay training. All 2,052 updates still sample the 16-window replay.
In contrast, `log.csv`'s `replayed_frac` describes only the newly collected window,
which has been zero throughout this arm. At 101 all 194 completed games ended in wins,
but at 102 there were 355 win endings and 167 clock endings (31.99%), and at
103 there were 268 win endings and 362 clock endings (57.46%). Clock endings
were 52.93% at 104, 53.25% at 105, 56.76% at 106 and 56.89% at 107.
At 108 there were 248 win endings and 328 clock endings (56.94%); at 109
there were 271 win endings and 309 clock endings (53.28%). At 110 there were
244 win endings and 309 clock endings (55.88%). At 111 there were 253 win
endings and 310 clock endings (55.06%). At 112 there were 232 win endings
and 341 clock endings (59.51%). At 113 there were 215 win endings and 330
clock endings (60.55%). At 114 there were 206 win endings and 352 clock endings
(63.08%). At 115 there were 217 wins, 351 clock endings (61.69%) and one
maximum-plies ending (0.18%). At 116 there were 231 wins and 302 clock endings
(56.66%). At 117 there were 568 endings: 225 wins and 343 clock endings (60.39%),
with no maximum-plies endings. At 118 there were 527 endings: 193 wins and
334 clock endings (63.38%), with no maximum-plies endings. At 119 there were
559 endings: 226 wins and 333 clock endings (59.57%), with no maximum-plies
endings. At 120 there were 543 endings: 220 wins and 323 clock endings (59.48%),
with no maximum-plies endings. At 121 there were 586 endings: 229 wins and
357 clock endings (60.92%), with no maximum-plies endings. At 122 there were
541 endings: 214 wins and 327 clock endings (60.44%), with no maximum-plies
endings. At 123 there were 525 endings: 212 wins and 313 clock endings (59.62%),
with no maximum-plies endings. At 124 there were 572 endings: 224 wins and
348 clock endings (60.84%), with no maximum-plies endings. At 125 there were
537 endings: 194 wins and 343 clock endings (63.87%), with no maximum-plies
endings. At 126 there were 559 endings: 215 wins and 344 clock endings (61.54%),
with no maximum-plies endings. At 127 there were 528 endings: 215 wins and
313 clock endings (59.28%), with no maximum-plies endings. At 128 there were
550 endings: 191 wins and 359 clock endings (65.27%), with no maximum-plies
endings. At 129 there were 555 endings: 218 wins and 337 clock endings (60.72%),
with no maximum-plies endings. At 130 there were 533 endings: 198 wins and
335 clock endings (62.85%), with no maximum-plies endings. At 131 there were
532 endings: 216 wins and 316 clock endings (59.40%), with no maximum-plies
endings. At 132 there were 542 endings: 184 wins and 358 clock endings (66.05%),
with no maximum-plies endings. At 133 there were 551 endings: 194 wins and
357 clock endings (64.79%), with no maximum-plies endings. At 134 there were
538 endings: 202 wins and 336 clock endings (62.45%), with no maximum-plies
endings. At 135 there were 552 endings: 193 wins and 359 clock endings (65.04%),
with no maximum-plies endings. At 136 there were 534 endings: 206 wins and
328 clock endings (61.42%), with no maximum-plies endings. At 137 there were
546 endings: 182 wins and 364 clock endings (66.67%), with no maximum-plies
endings. At 138 there were 544 endings: 203 wins and 341 clock endings (62.68%),
with no maximum-plies endings. At 139 there were 552 endings: 206 wins and
346 clock endings (62.68%), with no maximum-plies endings. At 140 there were
549 endings: 206 wins and 343 clock endings (62.48%), with no maximum-plies
endings. At 141 there were 513 endings: 185 wins and 328 clock endings (63.94%),
with no maximum-plies endings. At 142 there were 573 endings: 197 wins and
376 clock endings (65.62%), with no maximum-plies endings. At 143 there were
544 endings: 199 wins and 345 clock endings (63.42%), with no maximum-plies
endings. At 144 there were 566 endings: 221 wins and 345 clock endings (60.95%),
with no maximum-plies endings. At 145 there were 510 endings: 146 wins and
364 clock endings (71.37%), with no maximum-plies endings. At 146 there were
527 endings: 181 wins and 346 clock endings (65.65%), with no maximum-plies
endings. At 147 there were 519 endings: 173 wins and 346 clock endings (66.67%),
with no maximum-plies endings. At 148 there were 539 endings: 178 wins and
361 clock endings (66.98%), with no maximum-plies endings. At 149 there were
535 endings: 183 wins and 352 clock endings (65.79%), with no maximum-plies
endings. At 150 there were 507 endings: 157 wins and 350 clock endings (69.03%),
with no maximum-plies endings. At 151 there were 521 endings: 164 wins and
357 clock endings (68.52%), with no maximum-plies endings.
At 152 there were 556 endings: 169 wins and 387 clock endings (69.60%),
with no maximum-plies endings.
At 153 there were 527 endings: 187 wins and 340 clock endings (64.52%),
with no maximum-plies endings.
At 154 there were 540 endings: 185 wins and 355 clock endings (65.74%),
with no maximum-plies endings.
At 155 there were 531 endings: 177 wins and 354 clock endings (66.67%),
with no maximum-plies endings.
At 156 there were 515 endings: 178 wins and 337 clock endings (65.44%),
with no maximum-plies endings.
At 157 there were 537 endings: 176 wins and 361 clock endings (67.23%),
with no maximum-plies endings.
At 158 there were 546 endings: 191 wins and 355 clock endings (65.02%),
with no maximum-plies endings.
At 159 there were 529 endings: 189 wins and 340 clock endings (64.27%),
with no maximum-plies endings.
At 160 there are 526 endings: 192 wins and 334 clock endings (63.50%),
with no maximum-plies endings.
The steady-state outcome mix remains unestablished. Mean completed-game length
is 368.02 plies and the logged repetition diagnostic is 1.579%.
The 115 capped game's terminal row 18476 (step 18, environment 44, ply 999)
has `reply_ok=false`, `regret_ok=false` and `outcome_ok=false`, with capture
age 100 of 200. Its exclusion explains 568 labelled games versus 569 endings;
it is not a missing-game or label-formula fault. At 107 one of 566 endings
reached the 1,000-ply cap (0.18%). Its terminal row 164551 (step 160,
environment 711, ply 999) had `regret_ok=false`, consistent with the replay
labeller accepting only win or capture-clock endings. This explains that
iteration's 565-game ranking cohort without a missing-game or labelling fault.
Clock endings appearing in later windows confirm that
the first window's fresh-game initialization and unfinished longer games made
its outcome mix unsuitable for a claim that clock draws vanished. The observed
windows do not establish the steady-state outcome distribution of this arm.

The new trainer emitted the same startup AccumulateGrad stream warning at
04:11:13 UTC; stderr is 1,303 bytes. Its completed updates and finite checkpoint
show no observed capture failure. Healthy training is continuing.

The old arena watcher exited because `ckpt_000100_clock200.pt` matched its broad
filename pattern and failed numeric conversion. `arena/watch.sh` now accepts
exactly six-digit `ckpt_NNNNNN.pt` names; suffixes are ignored. Bash syntax,
a real idempotent `--once` scan and independent review passed. One replacement
watcher chain started at 04:14:45 UTC: launcher 27592, wrapper 31492, actual
watcher 13808. It remains active and successfully exported and uploaded 150.
The stderr retains the original filename error and two failed helper-launch
attempts; those helpers are gone. At 21:00:36 UTC it grew to 25,026 bytes
with the exporter’s existing ONNX deprecation and constant-folding warnings;
export verification and upload succeeded. Git Bash's `bin/bash.exe`
initializes the required PATH; the detached command explicitly executes
`/usr/bin/bash` and appends the existing logs. All tenth checkpoints through
150 are uploaded; 160 is next. Checkpoint 110 is the first arena export from
the new arm. Export began at 07:21:31 UTC and upload completed at 07:21:39 UTC
as `conv_g192p10_110`, with `replaced=false` and its upload marker present.
Its immutable checkpoint SHA-256 is
`0f3ad1ba34f648b2226e12bd570417ccb7668b8adfcc78c436f4af68dbd83578`;
CPU comparison at 110 verified that latest and immutable checkpoints matched
in every field.
The 51,174,804-byte ONNX SHA-256 is
`8d8acf15635d49d42e2f77d02cb541bd1904b9dfb12a5c3c32dd181011d55b46`.
Metadata identifies EMA iteration 110, 46 planes and 51 atoms, and matches
the independently checked checkpoint and ONNX hashes. Export parity maxima
were policy 3.1e-5, Q 4.6e-6 and value 8.9e-7. Six arena jobs started at
07:21:39 UTC. At 10:10 UTC all jobs had finished: 608 games, rating +176.8 with
half-width 29.7, and no remaining scheduler work. Its direct result versus
`conv_g160_120` is 12 wins, no draws and 12 losses in 24 games. Against its
parent `conv_g192p10_100`, it has 9 wins, 3 draws and 4 losses in 16 games.
The player is settled and not broken. The small direct samples and the combined
configuration changes do not isolate an RGSC effect or establish a strength gain.
Checkpoint 120 exported at 10:48:35 UTC and uploaded at 10:48:43 UTC as
`conv_g192p10_120`, with `replaced=false` and its upload marker verified.
Its 51,174,804-byte ONNX has SHA-256
`b9bf7910e1566bf7f8e854210383f5937a8604749b6d10ded111d4f26ceafc11`.
Metadata identifies EMA iteration 120, 46 planes and 51 atoms; its checkpoint
hash matches the immutable 120 hash above, and its ONNX hash was independently
verified. Export parity maxima were policy 1.9e-5, Q 6.0e-6, value 1.5e-6,
plies-to-end 4.2e-5, draw 4.7e-6 and WDL 1.8e-6. Six arena jobs started at
10:48:44 UTC. At 13:45 UTC all jobs had finished: 632 games, rating +205.3
with half-width 29.6, and no remaining scheduler work. Its overall record is
382 wins, 94 draws and 156 losses. Its direct result versus `conv_g160_120`
is 21 wins, 5 draws and 14 losses in 40 games. Against parent
`conv_g192p10_110`, it has 8 wins, 2 draws and 2 losses in 12 games.
The player is settled and not broken. The favorable direct point estimates
remain based on small samples; neither they nor the overall rating isolate
an RGSC effect from continued training and the combined configuration changes.
Checkpoint 130 exported at 14:11:50 UTC and uploaded at 14:11:59 UTC as
`conv_g192p10_130`, with `replaced=false` and its upload marker verified.
Its 51,174,804-byte ONNX has SHA-256
`d79cfc897ad8aca2f24f056da0128b377a07fc04b679cb8369782de0e9229194`.
Metadata identifies EMA iteration 130, 46 planes and 51 atoms; its checkpoint
hash matches the immutable 130 hash above, and both hashes were independently
verified. Export parity maxima were policy 2.2e-5, Q 6.3e-6, value 8.6e-7,
plies-to-end 5.0e-5, draw 4.6e-6 and WDL 9.2e-7. Six arena jobs started at
14:11:59 UTC. The player has parent `conv_g192p10_120` and is not broken.
Its completed evaluation was verified at 17:05 UTC: 628 games, rating +196.4,
half-width 29.6, with 382 wins, 79 draws and 167 losses. The direct record
against `conv_g160_120` is 2 wins, 2 draws and no losses in four games;
against parent `conv_g192p10_120` it is 3 wins, 4 draws and 1 loss in eight
games. The player is settled, all jobs finished and the scheduler has no work.
This final evaluation has been reported. The small direct samples do not
establish a strength gain; this arm's combined changes and continued training
also prevent attributing a causal effect to RGSC. Its rating uncertainty overlaps
the previously reported checkpoint 120 result.
Checkpoint 140 exported at 17:37:11 UTC and uploaded at 17:37:20 UTC as
`conv_g192p10_140`, with `replaced=false` and its upload marker verified.
Its 51,174,804-byte ONNX has SHA-256
`8f2991fdbead9687dfd6bcd64f1aa4f8c74e69e59b17b854c81e7907f664a74a`.
Metadata identifies EMA iteration 140, 46 planes and 51 atoms; its checkpoint
hash matches the immutable 140 hash above, and both hashes were independently
verified. Export parity maxima were policy 2.5e-5, Q 4.0e-6, value 1.2e-6,
plies-to-end 5.0e-5, draw 3.1e-6 and WDL 1.5e-6. Six arena jobs started at
17:37:22 UTC. The player has parent `conv_g192p10_130` and is not broken.
Its completed evaluation was verified at 20:59 UTC: 636 games, rating +214.8,
half-width 29.6, with 373 wins, 120 draws and 143 losses. Its direct record
against `conv_g160_120` is 8 wins, 4 draws and 4 losses in 16 games;
against parent `conv_g192p10_130` it is 17 wins, 4 draws and 7 losses in
28 games. The player is settled and not broken or retired. All jobs finished
and the scheduler had no work before checkpoint 150 arrived. The favorable
direct point estimates remain small samples, and rating uncertainty overlaps
the previously reported checkpoint 130 result. Neither these results nor this
arm's combined changes isolate a causal RGSC effect.
Checkpoint 150 exported at 21:00:34 UTC and uploaded at 21:00:41 UTC as
`conv_g192p10_150`, with `replaced=false` and its upload marker verified.
Its 51,174,804-byte ONNX has SHA-256
`a45570b8aca07dbcf94593de6907055f58862f74527e65ed401f73f743910d96`.
Metadata identifies EMA iteration 150, 46 planes and 51 atoms; its checkpoint
hash matches the immutable 150 hash above, and both hashes were independently
verified. Export parity maxima were policy 3.1e-5, Q 6.7e-6, value 8.6e-7,
plies-to-end 5.7e-5, draw 4.1e-6 and WDL 9.8e-7. Six arena jobs started at
21:00:43 UTC. Its parent is `conv_g192p10_140`; it is not broken or retired.
Its completed evaluation was verified at 00:31 UTC on 14 September: 631 games,
rating +205.0, half-width 29.5, standard error 15.04, with 386 wins, 74 draws
and 171 losses. The player is settled and all same-player jobs have finished;
the scheduler continues other evaluations. Against `conv_g160_120` it has
5 wins, 1 draw and 2 losses in 8 games. Against parent `conv_g192p10_140`
it has 6 wins, 2 draws and 8 losses in 16 games. The direct samples are small
and rating uncertainty overlaps the preceding evaluations. These results do
not isolate an RGSC effect or establish a strength change.
Checkpoint 160 exported at 00:24:01 UTC on 14 September and uploaded at
00:24:09 UTC as `conv_g192p10_160`, with `replaced=false` and its upload marker
verified. Its 51,174,804-byte ONNX has SHA-256
`3199b074be57cbe8cb07e7cd1840b2525edf9e4808bd2f343d8c5e762d5e34b6`.
Metadata identifies EMA iteration 160, 46 planes and 51 atoms; its checkpoint
hash matches the immutable 160 hash above, and both hashes were independently
verified. Export parity maxima were policy 3.1e-5, Q 5.4e-6, value 7.2e-7,
plies-to-end 6.9e-5, draw 5.0e-6 and WDL 1.8e-6. Its parent is
`conv_g192p10_150`; the player is not broken or retired and evaluation has
started. This ONNX parity check does not establish eager-probe versus compiled
production-actor parity.
The monitoring heartbeat now checks these process identities and settings,
uses audits starting at `max(101, N-6)`, and distinguishes archived iteration 101
from the new arm. The live trainer was not changed by this monitoring repair.

## Assessment

This assessment covers the original RGSC-enabled arm through its archived
iteration 101. Current-arm evidence is reported separately above.

Earlier audits established a reproducible weakness in rank-head selection
among played positions: in those windows it selected lower measured utility
than uniform selection within the same game, including when controlling for
actor version. Recent windows reversed this diagnostic, with eight consecutive
positive within-game intervals through 64; the interval includes zero at 65
and is positive again at 66–70, includes zero at 71, and is positive at 72–80.
That sequence of nine positive raw intervals ends with intervals spanning zero at 81–83.
Actor-adjusted intervals were positive for twenty-three consecutive iterations
at 59–81, then included zero at 82–83. At 84, the raw interval is negative
and the actor-adjusted interval narrowly excludes zero on the negative side.
This was the first negative raw and adjusted interval since 55. At 85, the
raw deficit narrows and its interval barely excludes zero, while the adjusted
interval spans zero again. At 86, the raw interval remains negative and the
adjusted interval still spans zero. Pooled selection has concentrated sharply
again at 86, with 91.97% of mass on one row and 95.11% in one game. At 87,
the raw interval includes zero again, while pooled concentration eases but
still places 50.80% on one row and 52.70% in one game. At 88, raw and adjusted
point estimates turn positive with intervals including zero. Pooled maximum
row/game masses ease further to 12.31%/38.52%. At 89, raw lift turns slightly
negative with an interval including zero; adjusted lift remains positive with
an interval including zero. Maximum pooled row/game masses rise again to
67.78%/76.18%. At 90, raw lift is positive with an interval still spanning
zero, while the actor-adjusted interval is positive again. Pooled maximum
row/game masses rise further to 74.85%/77.37%. At 91, the raw interval still
spans zero and the adjusted interval stays positive; pooled maximum row/game
masses fall sharply to 0.33%/2.44%. At 92, both raw and adjusted intervals
include zero; pooled maximum row/game masses rise again to 26.80%/93.42%.
At 93, raw and adjusted intervals are positive; this is the first positive
raw interval since 80. Maximum pooled row/game masses ease to 4.47%/7.97%.
At 94, the raw interval includes zero again while the adjusted interval stays
positive. Pooled maximum row/game masses increase to 48.33%/56.63%, while
actual buffer draws become less concentrated. At 95, raw lift turns slightly
negative and adjusted lift narrows toward zero; both intervals include zero.
Maximum pooled row mass eases to 19.88%, while maximum game mass rises to
81.48%. At 96, raw lift is essentially zero with an interval including zero;
the adjusted interval is narrowly positive. Maximum pooled row/game masses
ease to 5.36%/7.75%. At 97, raw lift is slightly negative and adjusted lift
is positive, with both intervals including zero. Maximum pooled row/game
masses rise again to 37.45%/86.20%. At 98, raw lift becomes more negative
with an interval barely including zero; adjusted lift is near zero, also
with an interval spanning zero. Maximum pooled row/game masses ease to
15.93%/23.72%. At 99, raw lift narrows toward zero and adjusted lift turns
positive, with both intervals including zero. Pooled selection becomes nearly
single-game concentrated: maximum row/game masses rise to 90.78%/99.95%.
At 100, raw lift is slightly more negative and adjusted lift narrows toward
zero, with both intervals still including zero. Maximum pooled row/game
masses ease sharply to 1.79%/12.25%. At 101, raw lift becomes more negative
with an interval barely including zero, while adjusted lift turns slightly
negative with an interval including zero. Maximum pooled row/game masses
ease further to 0.40%/3.66%, while actual buffer draws become more concentrated.
These windows show reversals in ranking diagnostics
and pooled concentration; they do not
establish a persistent regime. The earlier deficit did not prove that actual tree
restarts were worse, and the recent recovery does not independently validate
their benefit. Those require independent continuation measurements and a
controlled training comparison evaluated by the arena.

The evidence points to three interacting design concerns: the revised target
does not equal squared calibration bias; the listwise objective can reward
label variability and suppress corrective gradients after concentration; and
candidate selection and buffer priority use different quantities. Their
relative contribution in the trained network remains unresolved. A separate,
verified search defect repeatedly delayed immediate wins. It was corrected
before the frozen-checkpoint pilots, but does not establish the cause of ranking weakness:
the weakness survives the saved-trajectory shortening sensitivity below.

Recent diagnostic narrowing makes continued observation useful, but does not
establish that waiting alone resolves the target and objective concerns. The
previous run developed persistent negative within-game lift late in training.
At current-run iterations 49–50, both unadjusted and actor-adjusted lift intervals
included zero, but both returned below zero at 51. Iteration 52 had the first
positive unadjusted interval since the restart; its actor-adjusted interval
included zero. Both point estimates returned negative at 53–55, then approached
zero at 56. Iterations 57–64 had consecutive positive raw intervals, followed
by an interval spanning zero at 65, positive intervals at 66–70, and an interval
spanning zero at 71, positive intervals at 72–80 and intervals spanning zero
at 81–83 before returning below zero at 84–86, including zero at 87–92 and
becoming positive again at 93 and including zero at 94–101. At 59–81,
the actor-adjusted intervals were also positive, the first such sequence since the
restart, after spanning zero at 56–58. They include zero again at 82–83 and
are narrowly below zero at 84, then span zero at 85–89 and are positive again
at 90–91, include zero again at 92, are positive at 93–94 and include zero
again at 95, then are narrowly positive at 96 and include zero again at 97–101.
Pooled selection became concentrated
again at 60, eased at 61 and became substantially less concentrated at 62–63,
before concentrating again at 64–69, easing at 70–72 and becoming more
concentrated at 73–75, becoming less concentrated again at 76–80, then
increasing at 81, easing at 82–84, increasing again at 85 and becoming nearly
single-position concentrated at 86, then easing while remaining highly
concentrated at 87–88, before increasing again at 89–90 and easing sharply
at 91, then concentrating again at 92, easing at 93 and concentrating again
at 94. At 95, row concentration eases but game concentration rises; both ease
at 96, then both rise again at 97, ease at 98, increase sharply at 99 and
ease sharply at 100 and further at 101. Score stability
across games remains unresolved. The played-position diagnostic recovered
earlier, narrowed toward zero at 82–83, returned negative at 84 despite less
concentrated pooled selection, then narrowed at 85–87 and turned positive
at 88, slightly negative again at 89 and positive at 90–92, with all five
intervals including zero, then positive at 93 with an interval above zero,
still positive at 94 and negative at 95, then essentially zero at 96 and
slightly negative at 97, more negative at 98, narrowing again at 99 and
slightly more negative at 100 and again at 101, all with intervals including zero.
Its durability, cause and independently confirmed selection benefit remain unproven.

## What the paper does and what the amendment tried to improve

The paper defines regret as the remaining-trajectory mean of squared error
between the selected action's search value and the terminal outcome. Its
ranking objective emphasizes high-regret states, and candidate selection uses
argmax over played and searched states. Figure 7 compares the top 2,000
predicted states with uniform samples; it is not our pooled softmax readiness
gate. Its value-only selection ablation loses to ranking in its tested games.
That is a risk to investigate when proposing regression, not proof that
regression on a different target cannot work.
Source: [Tsai et al., sections 3 and 4.4, Algorithm 1](https://arxiv.org/html/2602.20809v1).

The released learner at the pinned commit also differs from our raw whole-batch
loss: it sorts regret labels, distributes them across 32 interleaved groups,
and weights each label by its squared within-group min-max normalization before
applying the exponential ranking objective. These are label strata, not game
groups. This code is a comparison reference, not a verified identification of
the exact implementation behind every published experiment.
Source: [released learner, commit 5c7f84f](https://github.com/rlglab/rgsc/blob/5c7f84f14690b8bf4bb1407a387ac15d973347fb/minizero/learner/train.py).

Our intended improvement was sensible: spend replay effort on systematic
prediction mistakes, rather than repeatedly selecting intrinsically variable
outcomes. For a fixed prediction `q`, with `mu = E[Z]`:

```text
E[(q - Z)^2] = (q - mu)^2 + Var(Z)
```

A well-calibrated position can therefore have large squared-error labels.
The amendment replaces each squared-error contribution with `2q(q-Z)`, then
takes its suffix mean. It retains the two-output regret head and ranking loss,
uses Gumbel sampling for candidate selection, and changes buffer priority after
two observations. See [approved specification, item 33](DESIGN.md) and the
actual implementations in [replay.py](py/conv/replay.py),
[train.py](py/conv/train.py) and [control.py](py/conv/control.py).

## What has been measured

The audit reconstructs completed games across saved windows and independently
recalculates both suffix targets from persisted selected-action Q values and
terminal returns. It checks label agreement and outcome perspective. This
validates reconstruction from the saved terminal return; it does not independently
prove that every saved terminal return came from a correct engine execution.

| Run and window range | Reconstructed labels | Formula mismatches above 1e-6 | Outcome-perspective mismatches |
|---|---:|---:|---:|
| g192p10, 0–24 | 4,789,756 | 0 | 0 |
| g160, 51–140 | 17,433,084 | 0 | 0 |

Maximum absolute label discrepancy was approximately `1.2e-7`. All audited
iterations reproduced their logged pooled lift to the diagnostic tolerance.
No swapped head outputs, ranking-loss sign error, or missing-label masking
mistake was found in code review.

| Selection diagnostic | Current run | Previous run |
|---|---|---|
| Within-game lift against the current target | Negative in every iteration 2–24; all 23 intervals below zero | Positive through 110, near zero at 111, negative in all 29 iterations 112–140; 28 intervals below zero |
| Final full-audit within-game lift | -0.01593, interval [-0.02193, -0.01011] at 24 | -0.01074, interval [-0.01334, -0.00830] at 140 |
| Final full-audit lift controlling actor-window offsets | -0.01181, interval [-0.01511, -0.00893] | -0.00674, interval [-0.00888, -0.00456] |
| Final full-audit pooled lift | +0.02923 | +0.51096 |
| Final full-audit pooled effective sample size | 2.99 played rows | 1.29 played rows |

Lift means rank-weighted target minus the uniform mean; negative is worse for
the quantity being measured. Intervals resample games and remain conditional
on the observed actor/opening population. Common actors and repeated openings
can add dependence. Pooled lift can look positive while within-game lift is
negative because pooled selection also shifts mass between games. It cannot
validate the within-game selection decision by itself.

Flattening current-run scores to temperatures 2 or 4 did not remove the
deficit. Selection against the original squared-error suffix was also negative
in current-run iterations 3–24 and 89 of the 90 audited g160 iterations.
That comparison does not independently prove failure: removing outcome
variance was the amendment's purpose, and the two targets can disagree.

Historical checkpoints 100, 110 and 120 had identical saved configurations;
the transition around 111–112 was not accompanied by skipped updates or a
rollback. This does not make training stationary: values, policies, shared
features, data distribution and labels all continued changing.

The final iteration in the full audit, iteration 24, took 14.73 minutes to
collect and 4.17 to train, with zero skipped updates and rollbacks. Policy
loss was 1.29543; fresh/replay value losses were 1.63685/1.76546. Tree prediction
error fell from 0.15549 at 21 to 0.10124 at 24. That mean-error improvement
does not demonstrate better ordering, and its logged definition includes
clipped buffer labels.

Rolling monitoring at 26 reproduces the log and still finds negative within-game
lift: -0.01230, interval [-0.01646, -0.00761]; controlling actor offsets gives
-0.01049, interval [-0.01311, -0.00774]. Its 21.0-minute iteration has no skips
or rollback. The preceding checkpoint permits an exact first-tree feedback
decomposition over 73 keys: signed error 0.16542 consists of clipping gap
0.16284 plus clamped-label error 0.00258. Predicted mean 0.07229 is close to
observed clipped mean 0.06971, whereas raw signed mean is -0.09313. Both
reconstruction checks pass. This demonstrates how the logged signed error can
rise without a comparably large mean error against the buffer's clipped label;
it does not establish accurate per-state predictions or recovered ranking.
The archived rolling diagnostic (`runs/conv_g192p10/killed_it101/rgsc_diagnostics.json`) records
the latest monitoring window separately from the full audits.

The last iteration before the switch, 31, used the original search. Its
within-game lift is -0.01944, interval [-0.02527, -0.01414]; controlling actor
offsets gives -0.01260, interval [-0.01736, -0.00858]. Temperatures 2 and 4
remain negative. Pooled lift is -0.22046, effective sample size 8.23 rows, with
97.74% of the mass in one game. The reconstruction matches the logged lift.
Collection/training took 16.98/4.20 minutes, totaling 21.18 minutes, with no
skips or rollback. Policy loss was 1.26274, fresh/replay value losses
1.62332/1.74935, and gradient norm 5.64036.

At 31, replay restarts were 48.88%, with 60.55% scored entries, 98.83% tree
entries and 496 distinct boards. Draw ESS was 270.03 and top-16 mass 15.41%.
Lost feedback affected 24/291 openings (8.25%) and 33/544 replay games; maximum
reuse was 11. First-tree signed error 0.11766 splits into clipping gap 0.10831
and clamped-label error 0.00935 across 61 keys. Lost-key and tree-error checks
both match. These diagnostics do not establish ranking recovery. The full
historical audit remains separate and unchanged. Under the original search,
iteration 31 omitted immediate wins
from 2,279 of 2,942 opportunities (77.46%).

The archived original-arm artifact covers windows 95–101. Its newest iteration
reproduces the archived log; early windows 95–97 are left-censored by this range.
The original complete-window audits remain the source of the historical
interval classifications. At 101, within-game lift is -0.00215, interval
[-0.00453, +0.00004222], and within-actor-game lift is -0.00049, interval
[-0.00251, +0.00141]. The raw interval includes zero at 94–101 after the first
positive interval since 80 at 93. Raw lift is more negative than at 100 and
its interval barely includes zero, as at 98; that boundary alone should not
define a new ranking regime. The adjusted interval includes zero at 97–101
after being narrowly positive at 96, and its point estimate turns slightly
negative. Repeated monitoring and dependent windows limit interpretation;
durable recovery is not established.
Temperature 2 at 101 gives -0.00112, interval [-0.00276, +0.00042]; temperature
4 gives -0.00032, interval [-0.00132, +0.00065]. Both include zero, as at 95–100,
following a positive temperature-4 interval at 94 and positive intervals for
both at 93. This intermittent sensitivity does not establish that flattening
scores reliably fixes selection. There are 505 reconstructed games and
127,992 completed rows, with no partial games excluded at 101.
Pooled lift is +0.01079, interval [+0.00365, +0.01731], with ESS 5646.48,
game ESS 117.84, maximum game mass 3.66% and maximum row mass 0.40%.
At 100 the corresponding concentration measures were ESS 891.24, game ESS 47.15,
maximum game mass 12.25% and maximum row mass 1.79%. The maximum saved rank
score falls from 2.0625 to 0.275390625, with median -8.375 at 101. Both row
and game concentration ease further, while pooled lift remains positive with
an interval above zero. Pooled stability and within-game ranking remain
distinct unresolved concerns; their earlier improvement did not persist
consistently. Pooled and within-game estimates measure different selection
effects; neither alone validates actual tree restarts. Actual buffer draw ESS
falls at 101 and top-16 mass rises, moving oppositely to pooled concentration,
as also happened at 94. The pooled diagnostic does not directly measure actual
buffer diversity; the 99.95% pooled game mass at 99 was not a measurement of
actual restart mass.
Iteration 52 was the first corrected iteration with positive unadjusted
within-game lift: +0.00474, interval [+0.00060, +0.00919]. Its actor-adjusted
lift was +0.00069, interval [-0.00205, +0.00333]. The positive point estimates
did not persist at 53–55, approached zero at 56, and became positive again
at 57–82 before turning slightly negative at 83, more negative at 84 and
narrowing again at 85–87, becoming positive at 88, slightly negative at 89
positive at 90–94, negative at 95, essentially zero at 96, slightly negative
at 97, more negative at 98, narrowing again at 99 and more negative at 100–101.
Of the seventy corrected iterations, twenty-three raw intervals are below
zero (32–48, 51, 54–55 and 84–86), twenty-three include zero (49–50, 53, 56, 65, 71,
81–83, 87–92 and 94–101), and
twenty-four are above zero (52, 57–64, 66–70, 72–80 and 93). The first sequence of eight consecutive
positive raw intervals ended at 65; its point estimate remained positive and
the interval returned above zero at 66–70, included zero at 71 and was positive
again at 72–80 before spanning zero at 81–83, returning negative at 84–86,
including zero at 87–92, becoming positive again at 93 and spanning zero at 94–101.
The raw intervals at 54–55 barely excluded zero;
small boundary changes should not be treated as a distinct training regime.
Controlling for actor offsets, lift narrowed from
-0.00712 at 45 to approximately zero at 49, widened to -0.00463 at 51, became
+0.00069 at 52, returned to -0.00241 at 53, narrowed to -0.00091 at 54,
and widened to -0.00286 at 55 before reaching +0.00039 at 56, +0.00126 at 57
and +0.00154 at 58. At 59 it reached +0.00576, the first corrected iteration
with an actor-adjusted interval above zero, followed by +0.00662 at 60 and
+0.00777 at 61, +0.00739 at 62, +0.00762 at 63, +0.00699 at 64, +0.00655
at 65, +0.00594 at 66, +0.00391 at 67, +0.00514 at 68, +0.00780 at 69 and
+0.00650 at 70, +0.00669 at 71, +0.00485 at 72, +0.00857 at 73, +0.00614
at 74, +0.00535 at 75, +0.00532 at 76, +0.00471 at 77, +0.00606 at 78,
+0.00599 at 79, +0.00490 at 80 and +0.00484 at 81 with
positive intervals. It narrowed to +0.00044 at 82 and +0.00038 at 83, with
both intervals spanning zero, then reached -0.00263 at 84 with a narrowly
negative interval, then narrowed to -0.00024 at 85, became +0.00022 at 86
and +0.00059 at 87, then +0.00239 at 88 and +0.00117 at 89, with all five
intervals spanning zero. At 90–91, adjusted lift reaches +0.00337 and
+0.00395, respectively, with both intervals above zero. At 92 it narrows to
+0.00178 with an interval spanning zero, then rises to +0.00452 at 93 and
narrows to +0.00344 at 94, both with intervals above zero. At 95 it narrows
to +0.00046 with an interval spanning zero, then rises to +0.00239 at 96
with an interval narrowly above zero, then reaches +0.00254 at 97 with an
interval spanning zero again. At 98 it falls to -0.00010 with an interval
still spanning zero, then rises to +0.00189 at 99 with an interval including
zero, then narrows to +0.00042 at 100 and turns slightly negative at -0.00049
at 101, both with intervals including zero.
The adjusted intervals at 56–58 also
span zero.
The recent improvement does not establish
durable recovery or its cause, particularly given the intervals and repeated
monitoring of dependent iterations.
Earlier narrowing from -0.02040 at 35 to -0.00719 at 38 did not persist at
39–40. Continued observation and independent confirmation remain necessary.

At 37, the within-game lift for exp(label) is +0.00468 despite negative raw
label lift. At 62, 65, 67, 77, 78, 82, 92 and 94 the signs disagree in the other direction:
exp(label) lifts are -0.00014, -0.00226, -0.00116, -0.00284, -0.00012,
-0.00406, -0.00154 and -0.00152 while raw-label point estimates are positive.
These are descriptive examples
of the two utilities disagreeing;
the exponential diagnostic has no reported confidence interval and uses game
groups, whereas the learner uses whole minibatches. It does not establish
that the trained head successfully optimizes its loss or explain the causal
source of the raw-label deficit by itself.

The first resumed collection starts fresh actor games, so its 12.19% restart
fraction and 4.63% replayed rows are transient comparisons with the preceding
steady run. Scored/tree buffer fractions are 56.84%/99.02%, with 496 boards,
draw ESS 301.83, top-16 mass 13.79%, clock left 71.34 and dominant clock-limit share
3.52%. No feedback was lost among 50 replayed openings; maximum reuse was two.
First-tree error 0.27162 comes from only eight keys: clipping gap 0.20304 plus
clamped-label error 0.06858. Both reconstruction checks pass. Do not infer a
steady-state regression or improvement from this first resumed window alone.

At 101, restarts were 48.89% and replayed rows 35.53%. Scored/tree fractions
were 93.16%/99.02%, with 474 distinct boards. Draw ESS fell from 327.57 at
100 to 192.67, while top-16 mass rose from 11.12% to 17.92%, indicating more
concentrated buffer draws despite less concentrated pooled diagnostics.
Clock left was 116.66 with dominant clock-limit share 2.15%. Lost feedback affected
4/278 openings (1.44%) and 30/483 restart games; maximum reuse rose from
sixteen to twenty. Logged tree error eased from +0.10609 at 100 to +0.06506,
also below +0.06973 at 96. Maximum buffer priority rose from 0.23167 to 0.50730.
The latest independent feedback reconstruction is iteration 101, using prior
checkpoint 100: signed error 0.06506 = clipping gap 0.07452 plus clamped-label
error -0.00946 across eight first-tree keys. Lost-key and tree-error checks
match the log. Prediction mean 0.03025 is below clipped observed mean 0.03971;
raw observed mean is -0.03481. Clipping exceeds the net signed error and is
partly offset by mean underprediction against clipped observations.
Prediction/raw Spearman is -0.11905 on these eight keys, with 62.50% negative
observations. Eight selected keys cannot establish population ranking or
individual calibration. This measures the regret prediction, not the rank
head, and has no reported interval or independent continuation validation.
At 96, signed error 0.06973 combined clipping gap 0.06159 with clamped-label
overprediction +0.00814 across fourteen keys, with about 88% from clipping.
At 91, signed error 0.04484 combined clipping gap 0.05737 with clamped-label
underprediction -0.01253 on eight keys. At 86, about 83% of the signed mean
error came from mean overprediction against clipped observations on seven
keys, with about 17% from clipping. Attributing every positive tree error to
clipping would therefore be incorrect. At 66, the near-zero signed error
reflected nearly cancelling clipping and underprediction terms, illustrating
why the signed mean alone is not a calibration test.
Iteration 101 collection/training took 1042.07/251.41 seconds
(17.37/4.19 minutes), totaling 21.56 minutes. It completed 2,052 updates,
with no skips or rollback and gradient norm 2.91845. Full-search rows were
54.69%, unchanged from 100. Collection time rose modestly from 1018.93 seconds;
this observational comparison does not isolate a search-fix speedup.
Policy/Q/value losses were 1.12997/1.47055/1.46049; fresh/replay value losses
were 1.53713/1.32347.
Checkpoint 101 was verified with 290 finite model and 290 finite EMA tensors,
290 finite optimizer states at step 209,304 and unchanged configuration.
Replay remains at the configured maximum of 16 windows, containing 3,145,728 rows.
Immutable checkpoint 100, used for the latest feedback reconstruction, has
205,582,141 bytes and SHA-256, reverified during the iteration 101 audit,
`9e299d14b4a2253384d3ecfde8559197034c6d34f39ac298ed1c61e13e37648d`.
The learning rate at 101, 0.0001717279, matches the configured cosine decay
after the hold through 40. All 417 immediate wins were selected at 101,
bringing corrected-window conversions to 34,585/34,585 across 13,762,560 rows
in 70 original-arm iterations, including the archived iteration 101. These
are historical measurements from trainer PID 76724; that trainer has since
been replaced by the current arm described above.
These are operating diagnostics, not strength measurements.

The existing watcher automatically exported checkpoint 40 and uploaded it at
07:11:37 UTC. Export verification measured maximum ONNX–PyTorch differences
of 2.3e-5 for policy logits, 2.2e-6 for Q and 7.3e-7 for value. The arena
started six jobs at 07:11:38 UTC, including direct matches against
`conv_g160_120`. At 09:36 UTC its arena evaluation had settled at 588 completed
games, rating +106.3 and half-width 29.7. All jobs had finished; the scheduler
had no further work. Its direct record against `conv_g160_120` was 4 wins,
2 draws and 10 losses in 16 games. These observations cannot isolate the search
fix's effect from the additional training iterations.

The same watcher automatically exported checkpoint 50 and uploaded it at
10:36:05 UTC. Maximum ONNX–PyTorch differences were 2.2e-5 for policy logits,
4.5e-6 for Q and 1.1e-6 for value. At 13:04 UTC the arena evaluation had
settled at 588 completed games, rating +124.2 and half-width 29.7. All jobs
had finished and the scheduler had no further work.
Its direct record against `conv_g160_120` was 12 wins, 7 draws and 13 losses
in 32 games. These results do not isolate a causal strength gain from the fix.
The same watcher automatically exported checkpoint 60 and uploaded it at
14:00:36 UTC. Maximum ONNX–PyTorch differences were 1.7e-5 for policy logits,
5.2e-6 for Q and 7.7e-7 for value. Six arena jobs started at 14:00:37 UTC. At
16:37 UTC, its arena evaluation had settled at 588 completed games, rating
+150.4 and half-width 29.6. All jobs had finished and the scheduler had no
further work. Its direct record against `conv_g160_120` was 8 wins, 2 draws
and 10 losses in 20 games. This small sample does not establish superiority,
and the overall rating does not isolate the search fix from continued training.
The same watcher automatically exported checkpoint 70 and uploaded it at
17:23:16 UTC. Maximum ONNX–PyTorch differences were 2.1e-5 for policy logits,
3.3e-6 for Q and 9.2e-7 for value. At 20:11 UTC, its arena evaluation had
settled at 604 completed games, rating +160.2 and half-width 29.6. All jobs
had finished and the scheduler had no further work. The direct record against
`conv_g160_120` was eight wins, five draws and seven losses in twenty games. This early sample does not establish
superiority or isolate the search fix's effect from continued training.
The same watcher automatically exported checkpoint 80 and uploaded it at
20:46:02 UTC. Maximum ONNX–PyTorch differences were 1.9e-5 for policy logits,
5.5e-6 for Q and 8.3e-7 for value. The 51,174,804-byte ONNX and successful
upload marker were verified. Six arena jobs started at 20:46:03 UTC. At
23:34 UTC, the arena evaluation had finished at 608 completed games, rating
+151.4 and half-width 29.4. All jobs had finished and the scheduler had no
further work. The direct record against `conv_g160_120` was six wins, two
draws and four losses in twelve games. This small direct sample does not
establish superiority, and the overall rating does not isolate the search
correction from continued training.
The same watcher automatically exported checkpoint 90 and uploaded it at
00:08:52 UTC on 13 September. Maximum ONNX–PyTorch differences were 2.5e-5
for policy logits, 4.8e-6 for Q and 1.6e-6 for value. The 51,174,804-byte
ONNX, its metadata linking EMA weights to the verified checkpoint hash, and
the successful upload marker were verified. At 02:59 UTC, the arena evaluation
had finished at 604 completed games, rating +164.1 and half-width 29.7. All
jobs had finished and the scheduler had no further work. The direct record
against `conv_g160_120` was six wins, two draws and eight losses in sixteen
games. This small direct sample does not establish superiority, and the
overall rating does not isolate the search correction from continued training.
The same watcher automatically exported checkpoint 100 and uploaded it at
03:31:44 UTC on 13 September. Maximum ONNX–PyTorch differences were 2.9e-5
for policy logits, 4.8e-6 for Q and 7.2e-7 for value. The 51,174,804-byte
ONNX, metadata linking EMA weights to the verified checkpoint hash, and the
successful upload marker were verified. Its ONNX SHA-256 is
`933996b1a7d8f58e3a545576edda754f38acf40a367d5964fd8e827984585e5c`.
Six arena jobs started at 03:31:46 UTC. At 06:13 UTC on 13 September, all
jobs had finished, the scheduler had no further work and the evaluation was
settled at 596 completed games, rating +178.0 and half-width 29.8.
Its direct record against `conv_g160_120` is seven wins, two draws and three
losses in twelve games. This favorable but small direct sample does not
establish superiority; the rating uncertainty also overlaps checkpoint 90's
settled result. Checkpoint 100 belongs to the original arm; later new-arm exports
must be kept distinct from any future unchanged control continuation.
The exporter remains active; checkpoint 110 is the next eligible export.

Evidence: current-run audit (`runs/conv_g192p10/rgsc_audit.json`) and
g160 audit (`runs/conv_g160/rgsc_audit.json`). Field definitions and commands
are in the [README](README.md#python-interface).

### Repeated-opening evidence

The audit uses only replay games completed wholly within one actor window.
For each repeated opening, it splits observations into a prefix and a disjoint
confirmation group, requiring at least two games in each. Selection takes the
top quarter separately within each actor. The stricter grouping also fixes
the selected opening action and full/cheap search mode.

The current run supplies 82 eligible state groups across 77 opening keys,
but only 15 action/search-matched groups across 14 keys. Its rank-based
confirmation intervals include zero. Millions of saved rows therefore do not
provide millions of independent calibration measurements.

The previous run supplies 1,607 state groups and 923 action/search-matched
groups. In the latter, prefix pair-product estimates select higher confirmation
values than uniform: lift +0.09828, opening-cluster interval
[+0.05668, +0.15588]. Rank selection gives +0.01390 with interval
[-0.02356, +0.05191]. This supports investigating repeated independent error
estimation; it does not establish a winning replacement. These saved games
were adaptively selected, share batch-level search-mode randomness, and are
not an IID experiment. They measure opening error, not remaining-trajectory
utility. Pooled rank correlations also mix actor score offsets and must not
be interpreted as controlled comparisons.

## Exact CPU findings and their implications

**The revised target is not squared bias.** Its expectation is
`E[2q(q-Z)] = 2q(q-mu)`. The following cases are exact calculations:

| Prediction q | Expected outcome mu | Squared bias | Expected revised target |
|---:|---:|---:|---:|
| 0.0 | 0.8 | 0.64 | 0.00 |
| 0.3 | 0.8 | 0.25 | -0.30 |
| 0.8 | 0.3 | 0.25 | +0.80 |

It can miss an error when Q is near zero and treat equally large underconfidence
and overconfidence very differently. This is a target-definition issue, not
an arithmetic bug in the persisted labels.

**Exponentiation can reward variability.** Two candidates can both have
expected label zero: one always has zero, while the other has -0.5 with
probability 0.75 and +1.5 otherwise. Selecting either exclusively gives expected
ranking loss zero. A 50/50 mixture gives approximately -0.087764, a better loss
without higher expected label. This is an exact two-candidate calculation using
the actual loss, not an approximation that replaces expected log loss with a
log of expectations.

**Wrong, concentrated rankings can receive almost no corrective gradient.**
For `L = logsumexp(gamma) - logsumexp(gamma + y)`, the gradient is
`softmax(gamma) - softmax(gamma + y)`. With correct labels `[1, 0]` and scores
`[-A, 0]`, the better candidate's gradient is:

| Score disadvantage A | Corrective gradient |
|---:|---:|
| 0 | -0.23106 |
| 20 | -3.54e-9 |
| 80 | -3.10e-35 |

A CPU tabular fitting experiment initialized at `[-80, 0]`, using 300 SGD steps
at learning rate 0.1, remained inverted under the current loss. Direct utility
MSE recovered the fixed labels to within `1e-10`. This isolates an objective
property; it is not a reproduction of the live optimizer or proof of neural
generalization. Entropy regularization alone also retains probability-weighted
logit gradients, so it is not a complete solution to severe saturation.

**Clipping restores positive bias to an otherwise unbiased pair estimate.**
For independent, fixed-policy residuals `r`, the statistic
`((sum r)^2 - sum(r^2)) / (n(n-1))` estimates `E[r]^2`. For two independent
fair residuals of -1 or +1, its expectation is zero, but its positive part has
expectation 0.5. Keep signed observations during statistical estimation and
training. Nonnegative replay probabilities require a separate, explicit mapping;
they do not justify claiming that clipped observations remain unbiased.

**A simple teacher suffix is not yet a validated replacement.** If an
outcome-independent teacher predicts mean residual `m`, the local observation
`2mr-m^2` has expectation `b^2-(m-b)^2`, where `b=E[r]`. It is useful only with
teacher error understood. Reusing the observed residual as its own teacher
reintroduces squared-error noise. Even an exact local teacher does not justify
averaging these observations over an outcome-dependent trajectory length:
the CPU fixture gives expected proxy -0.0625 versus actual mean squared bias
+0.1875. Independent confirmations at independently sampled future anchors
recover the intended positive quantity in that fixture.

**Conditioning and independence matter.** Opposite action-specific residual
means can cancel in a state-only mean. Combining observations from different
actors can mix opposite biases. Sharing full/cheap search randomization between
two supposedly independent continuations can bias their product. Separate
episode IDs alone are insufficient if random streams or outcomes are reused.

The exact cases and tabular experiment are implemented in
[control_audit.py](py/conv/control_audit.py) and
[test_control_audit.py](py/tests/test_control_audit.py).

## Proposed direction

Preserve both goals: find trajectories containing useful prediction mistakes,
and avoid treating intrinsic outcome variability as systematic error. Train
the utility prediction with a direct regression objective so that an example
can correct its prediction even when its current selection probability is low.
Use the same defined utility for candidate selection and buffer priority.

The candidate design is a conditional signed-residual predictor, followed by a
remaining-trajectory utility predictor trained or distilled from independently
validated squared-bias measurements. The primary candidate conditions residuals
on state, selected action and search mode, then integrates squared conditional
means into state utility before trajectory aggregation. Additional conditioning
on selected Q changes the target unless integrated out before squaring.
This is calibration under a specified policy and restart protocol, not
optimal-play error or guaranteed learning gain.

The implemented probe explicitly distinguishes state-policy, action/mode and
complete-root-context quantities. Independent CPU validation establishes
estimator construction; completed action/mode and root-context GPU pilots
establish execution and limited cost evidence. Confirmatory selection evidence
and real-feature head trainability remain pending. Their resource and acceptance
gates are specified below.

Requirements for a production proposal are:

- Independent data for fitting/calibrating residual estimates and checking
  utility; no same-outcome teacher leakage or unsupported suffix averaging.
- Signed error observations retained for estimation; explicit treatment of
  uncertain or negative finite-sample estimates at the sampling boundary.
- One utility definition throughout the head and buffer, with estimates
  refreshed for the current actor rather than switching definitions after
  two observations or accumulating incompatible actors indefinitely.
- An explicit exploration component so a zero or uncertain teacher does not
  prevent useful positions from ever receiving evidence.
- Held-out tests of the actual within-game/tree selection distribution,
  including concentration and calibration. Positive pooled lift is insufficient.

Changing temperature, periodically resetting the rank head, or applying
cross-entropy to softmaxes of the same noisy raw labels does not address all
these requirements. Direct regression remains a candidate, not an approved or
proven strength improvement. The CPU results establish mechanisms and test
infrastructure, not a trained replacement head.

## Experiment sequence and decision criteria

| Phase | Work | Status / resource |
|---|---|---|
| A | Label/tactical audits, mathematical fixtures, tabular loss comparison, probe bookkeeping, real-checkpoint compatibility | Complete on CPU within the stated scope |
| B | Corrected-search CUDA tests, then frozen-checkpoint pilot on real discovered tree/played candidates | 18 CUDA/graph cases passed; root-context and action/mode pilots completed and independently audited |
| C | Independent confirmatory selection measurements, including action-conditioned error | Pending; both pilots share one originating game and provide no independent replication or interval |
| D | Fit candidate heads with disjoint training and confirmation data | Planned; real-feature learnability is untested |
| E | Isolated controlled continuation tests and arena evaluation | Within the authorized investigation; required before production adoption of an RGSC replacement |

### A. Completed validation

The diagnostic and target audit passes 43 targeted cases. The immediate-win
correction passes 18 distinct search cases, including captured live-capacity
overflow. Independent review additionally checked 270 direct algebra/parity
cases for shortening sensitivity. The expanded probe passed 62 CPU cases in
766.66 seconds, covering all three estimands, root reproduction, rejection caps,
interruption/resume, full candidate-pool persistence and finite-sample completion
ranges. Eight additional return-support guard cases then passed, giving 70
distinct passing cases; the 27 completion-range cases were also rerun after
that guard was added. Lint and formatting passed. CPU execution intentionally
hid CUDA and used one Torch thread with a tiny evaluator. These tests do not
establish CUDA/compiled-actor parity or train a replacement head.
An expensive stochastic integration fixture was interrupted and replaced by
a bounded deterministic fixture; the interrupted case is not counted as a pass.
Stochastic correctness remains covered by separate fixtures and an independent
10,000-pair-per-estimand orchestration check against analytic expectations.
Real environment/search CPU tests use a tiny evaluator. All changed Python
files pass lint and formatting checks. The ten tactical cases affected by the
capture-safe test-fixture changes also passed again on CPU in 323.58 seconds.
Subsequent real CUDA validation and checkpoint pilots are recorded below.

Checkpoint 20's actual 12,815,948-parameter EMA loaded on CPU and returned finite
outputs through the real evaluator. Warm single-state evaluation took about
0.22 seconds in a three-call smoke check. This is compatibility and limited CPU
timing evidence, not GPU throughput or a completed real-model rollout experiment.

### B. CUDA validation and frozen-checkpoint pilots

The original trainer was stopped at 03:36:08 UTC on 12 September after atomic
checkpoint 31, its log and its persisted window agreed. The last reply masks
were clear and model/EMA tensors finite. The immutable
boundary checkpoint (`runs/conv_g192p10/ckpt_000031.pt`) has SHA-256
`a43555ab0f1d490ac2e6836a30ae1583bedc421e11e2e6576107b3e5bdad13c5`.
A CPU restore check found optimizer step 65,664, 12 replay windows (20–31),
2,359,296 rows, active RGSC heads and no warmup reset. The 127,879 unresolved
outcome/regret rows remain masked across resume.

All 18 search cases passed on the RTX 4070 Ti with Torch 2.10.0+cu126 and real
CUDA graph capture in 1.46 seconds. The first attempt exposed Python-list
index transfers inside two test callbacks; allocating their index tensors
before capture fixed the fixtures. Production search required no further edit.
Both eager checkpoint-31 pilots completed and were independently audited:

| Pilot artifact | Episodes | Complete pairs | Plies | Elapsed seconds | Matching attempts | Censored / capped pairs |
|---|---:|---:|---:|---:|---:|---:|
| Root context (`runs/conv_g192p10/rgsc_probe_31_root_context_overnight.json`) | 21 | 8 | 2,112 | 568.5999 | 0 | 0 / 0 |
| Action/mode (`runs/conv_g192p10/rgsc_probe_31_action_mode_overnight.json`) | 21 | 8 | 2,082 | 567.3834 | 56 | 0 / 0 |

Neither pilot censored an episode or discovery. Both used one originating
discovery game, one pair per measurement and one future anchor. Their discovery
seed, initial state, selected candidates and discovery outcome match; these are
two estimands on the same discovery, not independent replications. All three
learned selectors have negative measured opening and future lift relative to
uniform in that one game. Action/mode future lifts are -0.02446 for rank
sampling, -0.13227 for rank argmax and -0.15814 for predicted-regret argmax.
No game-bootstrap interval is available. These are runtime/protocol checks and
noisy single-game observations, not sufficient evidence to select an RGSC design
or estimate the variance across games. Zero observed missing pairs does not
establish population coverage. The action/mode pilot made 64 root preparations,
including eight outer roots and 56 matching attempts under its 128-attempt cap.

The uniform-selected opening illustrates the sampling uncertainty: its two
action/mode confirmation outcomes were both -1, while its separate base
continuation won. The base outcome is excluded from the confirmation label.
The single positive pair product does not establish that this opening has
large systematic error; outcome variability can produce it. More independent
games and repeated confirmation measurements are needed before choosing a head.

Training resumed at 04:05:03 UTC on 12 September as PID 76724, from the immutable
checkpoint above, after a 28-minute-55-second pause. Startup confirms iteration
32, CUDA execution and restoration of 12 replay windows. The process now loads
the validated immediate-win correction; the RGSC objective, head weights at
resume, buffer design and all original training flags were retained. Existing
stdout/stderr were archived within the run and the monitoring paths remain
`train.log` and `train.err`. Iteration 32 completed and saved its atomic
checkpoint at approximately 04:26:18 UTC: collection 1,010.14 seconds, training
256.39 seconds, total 21.11 minutes. All 2,052 updates completed with no skips
or rollback. Policy loss was 1.24644, fresh/replay value losses 1.62621/1.74225,
and gradient norm 5.48331. All 290 optimizer states reached step 67,716;
model and EMA each have 290 changed tensors and remain finite. Operational
configuration is unchanged. Training then continued into iteration 33.

The corrected window contains 196,608 played rows and 293 immediate winning
opportunities: all 154 full-search and 139 cheap-search opportunities were
admitted and chosen. There were zero omissions or retained-but-unplayed wins.
This verifies the correction in live self-play, without establishing a strength
gain or a material collection speedup. The second corrected iteration, 33,
completed at approximately 04:47:25 UTC in 21.11 minutes: collection 1,016.79
seconds and training 249.78 seconds. All 666 immediate opportunities (359 full,
307 cheap) were admitted and chosen, giving zero omissions across 959
opportunities in the two corrected windows. All 2,052 updates completed without
skips or rollback; the atomic checkpoint has optimizer step 69,768 and finite
model/EMA weights with unchanged configuration. Policy/Q/value losses were
1.24617/1.66578/1.65632, fresh/replay value losses 1.61934/1.74946 and gradient
norm 5.36593. Training continued into iteration 34. The initial two-iteration
resume verification is complete; overnight monitoring continues for every new
iteration.

PyTorch emitted an AccumulateGrad stream-mismatch warning at learner startup.
Warmup uses an explicit side stream, while capture uses PyTorch's internal
capture stream. A retained autograd node can cross these streams; the precise
trigger in this resumed process is unproved. Capture exceptions propagate in
the learner, so successful completion and the verified optimizer steps establish
that capture/replay did not fail. The warning is emitted once and cannot measure
the frequency of synchronization. Monitor subsequent training times; sharing
the warmup and capture stream explicitly is a candidate for isolated validation,
not a live change or a reason to interrupt this healthy run.

Resume restores model/EMA weights, optimizer steps and moments, replay windows
and the RGSC buffer; it is not an exact replay of uninterrupted training.
Actor games, trees/history and RNG states are not checkpointed. Unfinished old
games remain masked rather than receiving fresh-game outcomes. The remaining
GPU research waits until training releases the device. Resuming with the
validated search correction does not establish that RGSC ranking is solved.

Use [probe_control.py](py/conv/probe_control.py) for the remaining investigation,
retaining immutable checkpoint 31 as a reference. Do not use `latest.pt`.
Hash the search source in the protocol and use the same corrected search for
all head comparisons. Additional frozen references should include checkpoint
20, the final current-run checkpoint and g160 checkpoints on each side of the
ranking reversal. This requires no live-trainer restart.

For each fresh discovery game, the probe records four selections over the same
eligible played/tree occurrences: rank sampling, rank argmax, predicted-regret
argmax and uniform sampling. Repeated states retain their occurrence multiplicity.
Shared selected states reuse their confirmation measurements across methods.

Each selected opening receives continuation pairs for an explicit estimand.
Let `R = Q - Z`, `A` be the selected action, `M` the root-search mode, and `C`
the complete root-search context, which contains `A` and `M`:

| Estimand | Quantity at state s | Interpretation |
|---|---|---|
| `state_policy` | `(E[R given s])^2` | Opposite errors between actions can cancel |
| `action_mode` | `E[(E[R given s,A,M])^2 given s]` | Removes within-action/mode variation; primary candidate bias quantity |
| `root_context` | `E[(E[R given s,C])^2 given s]` | Removes future-outcome variance but retains root-search-context variability |

Without censoring these form a nondecreasing hierarchy by conditional
expectation. Their gaps are useful diagnostics; they are not interchangeable
labels. Finite signed pair estimates need not obey the hierarchy exactly.
Source review suggests restricted equality for this fresh-root probe: with
a deterministic evaluator and fixed fresh history, the balanced 128/16 and
16/4 schedules give a surviving action 30 and 6 visits respectively at full
candidate capacity. Other branches may change node IDs without changing that
action's canonical retained subtree. Ordinary reused trees/history and
unbalanced schedules invalidate these assumptions. Keep the estimands separate
until a bounded test compares every retained field after compaction and coupled
future trajectories under different root contexts, including CUDA execution.
The coarse pilot spent only about 12 of its 567 seconds on standalone root
preparations, so this possible simplification does not justify another pause.

Fine-root pairs reproduce the same selected action, Q and root-tree
fingerprint, then use independent future streams. Coarse pairs sample an
on-policy action/mode and independently rejection-match another root search
to it. The attempt limit creates missing measurements and must be reported.
For fixed mode and action probabilities `p(a)`, uncapped matching costs
`E[attempts] = sum_a p(a)/p(a)`, the number of supported actions. With cap K,
expected attempts are `sum_a [1-(1-p(a))^K]` and missing mass is
`sum_a p(a)(1-p(a))^K`. Concentrated policies therefore do not automatically
make unbiased coarse matching cheap. Rejection cost is a pilot gate, not an
assumed practical method for producing every training label.

A separate base continuation supplies uniformly sampled future anchors,
each receiving new independent pairs. Base and discovery outcomes never label
those pairs. Each anchor starts with a fresh tree and repetition history.
Thus the future endpoint averages restart-at-anchor error along a base path;
it is not automatically the error distribution of ordinary rows whose search
retains an earlier tree/history. Protocol identity includes checkpoint and
source hashes, EMA weights, configuration, device, precision and version.

Ply-capped discoveries yield no confirmations. Other incomplete measurements
are excluded and counted; resulting estimates can be biased by censoring.
Intervals resample originating discovery games, not the dependent methods or
individual pairs. The default two-discovery-game pilot is a functional/cost
check, not acceptance evidence. With four distinct selected states per game,
two pairs per measurement and one anchor, it uses at most 74 completed/censored
episodes, plus fine/coarse root preparation and any rejection attempts.

The checkpoint-160 pilot ran from 00:25:22 to 00:49:18 UTC on 14 September
after normal training completion, checkpoint/export verification and GPU release.
Worker PID 54344 exited successfully and released the GPU. Its protocol version
is 3, with identity
`4fc1533dcf952989e6e992d7dc6cca0bdf1504ad0286a05393f8402eef8f9f1e`.
It uses EMA, eager BF16 autocast and the recorded action/mode protocol. This is
a two-discovery-game cost/coverage pilot, not confirmatory or adoption evidence.
All nine source hashes remained unchanged during execution. It was launched
from the workspace root with the following command; this completed pilot must
not be launched again:

```powershell
python -m conv.probe_control --ckpt runs/conv_g192p10/ckpt_000160.pt --device cuda --estimand action_mode --out runs/conv_g192p10/rgsc_probe_160_action_mode_pilot.json --games 2 --pairs 2 --anchors 1 --max-root-attempts 128 --seed 1
```

The completed pilot (`runs/conv_g192p10/rgsc_probe_160_action_mode_pilot.json`)
contains 3,479,777 bytes, SHA-256
`a2dbe55cb833aa11063f77a5cef77ed153a6598e6c58f5f12081e8f6387cbae0`.
Elapsed time was 1,433.48 seconds (23.89 minutes), with 74 episodes, 5,858
plies and 443,600 scheduled simulations. All 32 planned independent pairs
completed; there were no outcome, discovery or root-attempt caps. The 230
root preparations comprise 32 outer roots and 198 matching attempts, taking
57.97 seconds, 4.04% of elapsed time. Total recorded search/play time was
1,420.34 seconds; the remaining 13.14 seconds include orchestration and output.
Two episodes ended decisively and 72 by the capture clock. This is one measured
pilot cost, not a reliable runtime prediction for different states or protocols.

Parent and independent reviewer CPU reconstructions verified every signed product as the product of its
two independently seeded residuals, each residual as Q minus outcome, matching
state/action/mode identities, all requested denominators, means, costs and the
protocol/checkpoint/source hashes. Each selection has two opening pairs, one
independent base trajectory and one uniformly sampled anchor with two fresh
pairs. There are four distinct selections per game and two originating games.
With no missing planned measurements, every finite-sample completion range
collapses to its observed mean; it still omits sampling uncertainty.

The independent audit also checked every rejected root proposal: 198 proposals
included 166 rejections, with a maximum of 23 attempts for a match. Selected Q
matched exactly within all 32 independent-root pairs, while raw tree hashes
matched in only one pair. This supports investigating the restricted fresh-root
equivalence described above; it does not establish equality of retained subtrees
or future trajectories.

| Method minus uniform | Opening mean lift | Future mean lift |
|---|---:|---:|
| Rank sample | -0.0008167803 | -0.0000001896 |
| Rank argmax | +0.0049610938 | -0.0000000061 |
| Regret argmax | -0.0008146981 | +0.0000206817 |

These are pilot observations, not confirmed method rankings. The opening
rank-argmax differences are +0.00992218 in discovery 0 and +0.0000000048 in
discovery 1. The future regret-argmax differences are +0.0000412935 and
+0.0000000699. Their two-game bootstrap intervals exclude zero simply because
both observed game differences are positive; resampling two games cannot
represent unobserved game variation. They do not establish superiority.
The signed-pair observations range from -0.00017541 to +0.02328694; retaining
their signs is essential even though expected conditional squared bias is
nonnegative.

Rank-sample selections have only 2 and 7 capture-clock plies remaining, and
their independent base trajectories last 2 and 7 plies. Their opening
measurements are approximately 2.12e-10 and 3.20e-8. The second discovery's
other opening measurements are also around 1e-8. Thus tiny measured errors
in these selected near-clock states cannot support a general conclusion about
head quality, future learnability or the useful scale of an improvement.
Two games and one anchor per selection are inadequate for a dependable
variance estimate or confirmatory sample-size choice. A separate, fixed
variance-calibration stage is needed after resolving production-actor parity;
it must remain separate from fresh confirmatory seeds and any fitted head's
held-out evaluation.

Discovery 0 completed uncensored in 399 plies and 96.14 seconds. Independent
CPU review used one in-memory progress snapshot and validated its 39,701
ordered occurrences, 24,901 distinct states and 399 offer batches. There are
399 played and 39,302 tree occurrences, with 14,800 repeated state occurrences.
The decoded payload, saved batch order, checksum and replayed snapshot match.
An independent vector RNG/argmax calculation reproduced all four selections
and their probabilities exactly. Raw payload SHA-256 is
`67cf8f5e4484fc1d6d7ec660cc6d701c6da7080fa997b60aa3f66077a89168a0`.
Its 7,212,581 raw bytes compress to 672,102 zstd bytes and 896,136 base64
bytes. CPU validation took 1.24 seconds without CUDA initialization, edits
or new artifacts. This validates recording and replay of the candidate inputs;
it does not validate a new head or independently confirm selection utility.

Discovery 1 also completed uncensored: 340 plies in 75.03 seconds. Separate
CPU validation used one progress snapshot and checked 28,616 ordered occurrences,
15,890 distinct states, 340 offer batches and 12,726 repeated state occurrences.
Its sources comprise 340 played and 28,276 tree occurrences. Checksum, frame
size, ordering, helper replay and the independent four-selector oracle all
match exactly. Raw payload SHA-256 is
`6619383d1886bdc06070b92ad3583ab971bcce831990be8d1ef8449d1b641a0e`.
Its 4,763,364 raw bytes compress to 433,071 zstd bytes and 577,428 base64
bytes. CPU validation took 0.856 seconds, with no CUDA initialization, edits
or scratch; discovery 0 was not rechecked. All eight selected states across
the two games are tree candidates, with four distinct selected states within
each game. This is recording/replay validation, not an RGSC efficacy result.

The production-actor comparison tests three factors identified by source
inspection: eager versus compiled network execution, production's
enabled CUDA matmul TF32 versus the pilot's disabled setting, and production
batch size 1,024 versus the probe's batch size one. Both use BF16 autocast and
search CUDA graphs. A matched comparison must force the same full/cheap mode
and root random tensors or RNG state within each batch shape. Production uses
Torch mode draws and calls `search.sample_node()` on the search RNG after
each root; the probe uses NumPy mode draws and omits that sampling draw.
Equal master seeds therefore do not imply equal whole-game randomness. These
differences require controlled checks; they do not by themselves prove a
marginal-policy error or explain the RGSC ranking deficit.

The reviewed parity experiment compares three paths with the same immutable
EMA weights: A, eager BF16 with CUDA matmul TF32 disabled; B, eager BF16 with
TF32 enabled; C, production `compile_net` with BF16 and TF32 enabled. Pin
backend flags before compilation and separate compile/warm-up time from execution.
Freeze 16 distinct states before comparing outputs, spanning played/tree
sources, early/late plies, near-clock states, score ranges and actual method
winners, including the ordinary initial position. Preserve exact state fields.

Compare evaluator outputs at batch sizes one and 1,024: exact legal masks/counts,
finiteness, Q/V error, legal-policy total variation and both control scores.
Compare A/C across 64 matched fresh root contexts: 16 states, two fixed seeds
and full/cheap modes. Check played actions/Q, improved-policy differences,
admitted/surviving actions and candidate-state multisets. Repeat several
same-path contexts, extend four cases for two plies with mode switching, and
add one heterogeneous batch of 1,024 roots per mode. Keep one search arena
resident at a time. Finally rescore every distinct saved candidate state with
C, restore occurrence multiplicity and offer batches, and compare rank-softmax
total variation, both argmax winners and the original coupled rank sample.
Fixed-pool rescoring measures score sensitivity; root comparisons separately
measure changes in candidate availability.

Nonfinite outputs, changed legal/tactical results or failed same-path
reproduction block transfer. Small head errors alone do not validate transfer;
material selection-probability changes or action flips with clear score margins
need investigation. Passing these bounded checks establishes applicability on
the checked states and protocols, not global trajectory equivalence or RGSC
efficacy. If transfer remains ambiguous, collect subsequent utility evidence
directly with compiled production settings under a new protocol fingerprint.
The reusable [parity command](py/conv/probe_parity.py) passed 20 focused CPU
cases in 24.25 seconds, including real-engine clock termination and retained-tree
full-to-cheap switching with a tiny evaluator. Lint and formatting pass.
Independent source review found no remaining material blocker. All nine original
production/probe source hashes were unchanged. At that run, the command source hash was
`5b5c12eb30cfabe9a346c8e4d6e8ef7918e2dc3cd1debae576b65c02d67a139a`.

The real pilot parser and panel planner passed CPU validation. The fixed plan
contains 191 units: A has 74, B has two and C has 115. These comprise six
evaluator units, 128 individual matched roots, four repeats, eight two-ply
switch units, four 1,024-root units and 41 full-pool rescore chunks. Panel hash
is `7e08f2e4c55d9ede015f0c150335a05b3375f423d40c41c993813b969ca0601f`;
heterogeneous-batch hash is
`1c27d24d32a310a951c106b854ad5427f487885e0f684a2c579f5410641cc41d`.
Final survivor scores and infinity masks are recorded; they do not identify
earlier elimination margins. A/C search differences combine compilation and
TF32 factors, while B isolates the evaluator flag change.

After a fresh GPU ownership check, the command ran from 01:18:21 to 01:22:38 UTC
on 14 September as PID 62816, with output
rgsc_parity_160.json (`runs/conv_g192p10/rgsc_parity_160.json`).
It exited successfully after all 191 units in 252.64 seconds (4.21 minutes).
The completed artifact has 21,514,705 bytes and SHA-256
`47f8365eb6c08c13f4e1aeb3dd257b89b38b2bd58e2d5bb241014b90c2da602e`.
Parent reconstruction verified every unit signature, finite saved arrays,
all ten source hashes and every saved array in the four same-path repeats.
The repeats are bitwise exact. A and B evaluator arrays are also exactly equal
at both batch shapes, so enabling TF32 alone changed none of these eager outputs.
Independent review reconstructed all 155 saved comparisons, including the full
pool selectors and probabilities, and validated all 199 array payloads plus the
checkpoint/configuration identities. All 4,244 recorded root actions match the
maximum saved final score among active survivors. No artifact defect was found.

Compilation changes outputs despite identical legality. The A/C evaluator
comparisons have the following maximum absolute errors and policy total variation:

| Evaluator batch | Q error | V error | Rank error | Legal-policy TV |
|---|---:|---:|---:|---:|
| 1, over 16 panel states | 0.0390625 | 0.00073242 | 0.125 | 0.00552934 |
| 1,024 heterogeneous states | 0.04296875 | 0.0078125 | 0.25 | 0.02621237 |

A/C actions agree in 63 of 64 individual roots and 1,018 of 1,024 roots in
each batched mode. All compared root move mappings and tactical arrays agree.
Initial physical candidate sets, ignoring order, agree in 61 of 64 individual
roots and 1,989 of 2,048 batched roots; active initial sets agree in 62 and
2,007 respectively. Surviving sets agree in 62 and 2,035. Ordered array equality
is stricter and must not be described as a candidate-set change. Full tree-state
multisets agree in 45 of 64 individual roots, 901 of 1,024 cheap batched roots
and 280 of 1,024 full batched roots. A/C raw tree hashes need not match when
numerical fields differ. All four two-ply mode-switch pairs retain the same
states and actions through both measured steps.

Compiled rescoring preserves every candidate occurrence and original selection
RNG. Rank-probability TV is 0.01636130 for discovery 0 and 0.01692858 for
discovery 1. Both rank samples and both rank argmax choices remain identical.
Discovery 0 retains all four choices; discovery 1 changes its predicted-regret
argmax. Fixed-pool stability does not establish stability of the pools generated
by different search trajectories. Subsequent fresh utility collection should
use compiled production settings under its own recorded protocol. These results
show bounded numerical/search differences, not an explanation of the live
ranking deficit or an RGSC repair. Batch shape and retained history still limit
transfer from a scalar fresh-root protocol to ordinary actor trajectories.

The individual action flip is a near-tie: panel 13's cheap root changes action
201 to 265, with Q zero in both paths and final margins 0.00310 and 0.00129.
Some batched differences amplify through search. Full-batch row 811 changes
512 to 417, Q -0.04530 to -0.09702, with target TV 0.784216 and final margins
2.00749 and 2.34749. Its survivor sets change from [417,512] to [417,255];
actions 512 and 255 exchange 30/14 visits for 14/30. Row 770 keeps action 121
but changes Q from 0.394824 to 0.237295 despite 30 root visits in both paths.
All 13 action flips retain the same initial admission set. These observations
identify sensitivities, not their first divergent leaf or elimination decision,
and do not prove a search defect. The matched measured search totals were about
2.01 times faster compiled for individual roots and 2.68 times faster for batch
roots; these exclude parts of end-to-end probe work and do not predict rollout
throughput.

The bounded stress comparison completed on 14 September at 02:01:09 UTC,
using explicitly identical saved Gumbels across compiled batch-one and
batch-1,024 roots. Equal generator seeds alone do not supply that coupling.
The original heterogeneous batch context, selected states, modes and seeds
were fixed before the new outputs. This targets observed sensitivity; it is
not a representative accuracy sample.

The follow-up has a fixed 12-unit plan: two noise-generation units, six full-root
units and four cheap-root units. Regenerate the original full-batch noise once
and retain its exact bytes; repeat the original heterogeneous batch and scalar
rows 770/811 twice each. For the near-tie case, regenerate the original cheap
batch noise and replace only row 13 with independently regenerated original
scalar-case noise, then repeat that constructed batch and scalar row twice.
Its cheap batch is explicitly constructed, not an old-batch replay. A local
search subclass substitutes only `_start` noise and forwards the existing search
logic, validating the supplied and persisted tensors on warm-up and measured
calls. Request/source identity and noise dependencies are preserved on resume.
The 34 focused CPU parity tests passed in 23.96 seconds. After the final bitwise
noise and four-call validation guards, their two affected cases passed again;
these are 34 distinct cases, not a second complete run. Lint/format passed,
and Peirce independently approved the final source before GPU execution.
Source SHA is
`94cbe93fab33cd6ff82beff049c600b9f9325ec596f41f0c9d4cdd898bb0d7c7`.
All 12 units completed successfully in 59.90 seconds; the worker exited normally.
The new artifact is
rgsc_parity_160_batch_shape.json (`runs/conv_g192p10/rgsc_parity_160_batch_shape.json`),
7,586,617 bytes, SHA
`b6632612865ee12be0348479d80be0e2ff06f72e4b8d1839af88056c6b4355e1`.
Parent validation matched all ten source hashes, the immutable checkpoint,
the original artifact and its source identities, configuration and runtime.
The artifact has 21 comparisons, no remaining units and no execution error.
Astra independently checked the raw numerical comparisons. Peirce reconstructed
all 21 comparisons and final margins, validated all unit/payload and request
bindings, and checked all 40 recorded `_start` calls. Stored Gumbels plus root
log probabilities reconstruct initial scores bitwise on all 4,102 measured rows.
All five repeat pairs match every saved array bitwise. Neither review found an
artifact, noise-coupling or source-correctness defect.

Both repeated full batches reproduce the earlier C batch outputs exactly.
The constructed cheap batches reproduce the 1,023 unmodified rows exactly;
their replaced row 13 is a separate controlled context. Each scalar context
also repeats exactly. With identical injected noise, the cross-shape results are:

| Selected state and mode | Compiled batch one | Compiled batch 1,024 | Difference |
|---|---|---|---|
| Row 770, full | Action 121, Q 0.23690389 | Action 121, Q 0.23729451 | Q 0.00039062; target TV approximately 2.72e-11 |
| Row 811, full | Action 512, Q -0.04589030 | Action 417, Q -0.09701869 | Q 0.05112839; target TV 0.78284562 |
| Row 13, cheap, constructed noise context | Action 265, Q 0 | Action 201, Q 0 | Final margins 0.00128794 and 0.00066519; target TV 0.00299995 |

Row 811 starts with the same admitted candidates, but its final survivor sets
are [417,512] and [417,255], with final winning margins 1.92986 and 2.34749.
This difference persists under exact noise coupling and repeats. The recording
still does not locate the first divergent leaf or elimination round. Row 13
retains the same initial admission set and the same two final actions, changing
their order near a tie. Compiled
batch-one execution therefore cannot be treated as interchangeable with
production batch-1,024 search. This is evidence about the experiment's execution
context, not an established cause of the live ranking deficit or a measured
strength loss. Set the next collector's execution context explicitly before
generating independent utility labels.

The next protocol-4 collection is explicitly a **compiled batch-one development
and learnability calibration**. Its independent continuation streams remain
valid for that execution protocol. Its variance estimates size that development
design; favorable results cannot authorize production RGSC adoption. Both
reviewers recommend ending this numerical investigation and advancing to the
target/objective experiments under that stated scope. Fresh held-out utility
confirmation from a validated batch-1,024 collector remains required before
adoption. Capturing features at batch 1,024 does not turn batch-one continuation
labels into production labels.

Measured median compiled fresh roots cost 27.46 milliseconds cheap and 162.78
milliseconds full at batch one, versus 1.174 and 8.859 seconds at batch 1,024.
Those are actual search-shape costs, not measured padding costs or complete
rollout throughput. Scalar padding would spend nearly an entire large neural
batch on one sampled root and would still need search-context validation. A
future populated batch collector can amortize that work; paired members need
separate future mode streams or independently seeded waves, and uncertainty
must retain shared wave dependence. The next small protocol-4 smoke uses one
discovery, one base, one anchor and one pair per distinct selected state,
future-only: at most 13 episodes before duplicate selections. Measure its
orchestration and storage costs before committing to the development budget.

The development feature extractor passed 24 focused CPU tests in 2.55 seconds,
lint/format and independent source review. Its original C pool mapping contains
40,733 global states, 40,791 context-specific feature rows, 41 chunks and 68,317
occurrences. It preserves original chunk order and repeat-last padding. The
first real CUDA attempt stopped at its capture guard: every one of the 41
unhooked chunks exactly matched saved C scores, with no graph breaks, but the
first capture buffer was not fully written. No usable feature cache or head
reconstruction resulted. The failed artifact is
rgsc_features_160.json (`runs/conv_g192p10/rgsc_features_160.json`),
SHA `e641fdf15b4e998b69c96ed2080c3391303462418aec47817e08399348f2c8f7`,
83,665,324 bytes, elapsed 25.38 seconds. Its source SHA is
`e5d8a9edbb5153d485ca274ed5ac62306cf0dfcef7fbe8627b2c6f6cc1766f6e`.
Installed PyTorch 2.10 defaults to `skip_nnmodule_hook_guards=True`, which can
ignore a hook added after compilation. Both reviewers identified this cause.
The extractor now scopes hook guards from before its first compilation through
reconstruction. A CPU regression reproduces the old failure and verifies hook
attachment, refreshed buffers and removal with guards enabled. All 26 CPU tests
passed in 2.67 seconds, with lint/format and independent review passing.

The guarded GPU retry captured all 41 chunks but failed the exact numerical
gate in 36.64 seconds. Its artifact is
rgsc_features_160_guarded.json (`runs/conv_g192p10/rgsc_features_160_guarded.json`),
87,108,198 bytes, SHA
`2d89e538099e26d3813ed2880efb942e48ec5cc0adcc8dc95884302eccf1ff8c`.
Source SHA is `bae6d8ca9249730d6a809f7497821699015cb58d4171c4e4cb39579d16488967`.
Parent validation matched all 11 source identities, the checkpoint and source
parity artifact. Unhooked outputs still match all saved C scores exactly, and
there are no graph breaks. Hooked outputs change across all six numerical
evaluator fields; legal actions and counts remain exact. Rank differs at 3,172
elements with maximum 0.25, regret at 10,052 with maximum 0.001953125, and
selected head reconstruction differs from the unhooked scores at 13,224
elements. These counts include padded rows. Peirce independently verified that
all 83,968 reconstructed score values match the hooked scores bitwise; feature
storage and head reconstruction are exact for the instrumented graph. All
81,582 valid unhooked scores match saved C, and all 287 reported failures
reproduce. This is a change in compiled execution caused by instrumentation.

A CPU recording-backend reproduction warms batch one then batch eight. The
unhooked graph has a symbolic batch dimension; copying into the fixed capture
buffer specializes the hooked graph to literal eight. Removing only the shape
check does not prevent specialization because the copy also constrains shape.
Marking the capture buffer's batch dimension dynamic preserves a shared symbolic
batch while retaining shape/dtype/device checks. The approved bounded correction
also preserves a dynamic standalone-head input and adds a graph-shape regression.
The corrected extractor passed all original exact gates on the GPU. All 27
focused CPU tests passed in 3.08 seconds, lint/format passed, and Peirce approved
the dynamic-batch correction before this third GPU attempt. The successful
artifact is
rgsc_features_160_dynamic.json (`runs/conv_g192p10/rgsc_features_160_dynamic.json`),
87,040,179 bytes, SHA
`5107d6652053f3a753d47e23121a630f38ea6b05dad1e2667fe3d766fefec109`.
Its source SHA is
`e58db90abdf8531a528f8b4ca610480be75009a54455faa346fddc0cb2f8c952`.
The worker ran from 02:19:57 to 02:20:50 UTC on 14 September and exited normally;
recorded extraction/setup time is 41.71 seconds. All 41 chunks completed with
no graph breaks or numerical failures. Parent verification matched all 11
source hashes, checkpoint and source artifact. The cache contains 40,733 global
states, 40,791 valid context-specific feature rows and 68,317 occurrences.
Peirce independently validated all 81,582 valid saved-C scores and all 83,968
unhooked/hooked/reconstructed control scores bitwise, all 10,747,904 finite BF16
feature values and their lossless round trip, and the occurrence/state/padding
mappings. All 1,251 repeated mapped rows have identical feature bits: 1,193
padding rows and 58 cross-family shared states. The other evaluator fields total
81,826,816 exact elements in stored comparison metrics; their full raw arrays
were not retained, so those metrics were checked without a second raw-array
reconstruction. The source freeze was released after this audit.
Preserve both earlier failures as
evidence of the checks and their corrections. This validates feature extraction;
it does not supply utility labels, fit a head or modify training weights.

Use separate pilot artifacts for the other two estimands, with the same
discovery seeds and settings. The output must be new; `--resume` continues
the same artifact only when protocol and settings match. Completed episodes
and root attempts are saved atomically; an interrupted unit is repeated.
Never launch two workers against one artifact. Record wall time, episode lengths,
censoring, simulation budgets and variance of method differences. The completed
single-game pilots took about 9.5 minutes each; this limited observation is not
a reliable estimate for larger samples. Training's batched throughput is not a
defensible substitute for measuring this probe's costs.

### C–D. Confirm selection utility and trainability

Before collecting the confirmatory sample, choose a minimum useful lift and
fixed sample size using a separate variance-calibration stage and measured cost.
The two-game pilot cannot supply a dependable variance estimate. Use fresh seeds and avoid
repeatedly extending the sample until significance appears. Define whether the
primary endpoint is opening or future utility. The provisional primary endpoint
is future `action_mode` utility under the explicitly recorded restart-at-anchor
protocol; opening, state-only and fine-context measurements explain mechanisms.
Confirm that this endpoint matches the eventual teacher/training data protocol
before a head comparison. Complete-case intervals cannot establish superiority
if censored or rejection-capped actions could reverse the ordering. Under the
current bounded-return protocol, true utility lies in [0,4]; signed pair
observations lie in [-4,4]. Use conservative missing-data sensitivity, including
uncertainty in the missing mass, rather than dropping rare actions or assigning
zero utility to unfinished measurements.

The next discretionary allocation is a representative head-fitting development
screen, rather than a corpus consisting only of the existing selectors' winners.
Winner-only data would cover at most 96 selected states in 24 games and omit
most alternatives a new head might choose. No 24-game calibration has launched
or supplied outcomes. The old two-pool feature asset is for mechanics and
representation checks, not independent labeled training examples.

Preassign 24 fresh development discoveries to 16 training, four validation and
four locked test games. In each full pool, sample 16 occurrence IDs uniformly
without replacement before utility labeling. Preserve repeated-state
multiplicity and equal game weights. This gives 384 occurrence slots: 256
training, 64 validation and 64 test. Share measurements only for identical
sampled states within one discovery family, retaining all occurrence weights.
At each sampled state use one independent base, one IID uniform anchor and
one independent action/mode pair. Record both current excess from that complete
base and independently confirmed future utility. This first stage requires at
most 1,152 labeling episodes plus 24 discoveries, before duplicate-state savings;
root rejection preparations are additional.

After freezing the four fitted heads, score every occurrence in each of the
four test pools. Freshly confirm the actual winner of each new head, the
untouched rank/regret winners and uniform selection, using two bases, two
anchors per base and two pairs per anchor. At seven distinct choices per game,
that is at most 28 selected states and 504 further episodes. The complete
development screen therefore has a ceiling of 1,680 episodes, plus root
preparations and matching, before duplicate-state savings. It has not launched.
Fix a wall-time/resource ceiling using the one-game smoke before starting.
Resource exhaustion leaves an incomplete experiment, not statistical success.

This is a discretionary development screen, not a powered confirmation study.
Its 256 training slots are small relative to the 33,025-parameter scalar head;
failure cannot establish that the target is unlearnable. Four locked test games
cannot establish production utility or reliable absence of benefit. Keep the
production batch-1,024 utility and controlled strength gates separate.
Prefer fresh batch-one features for this batch-one labeling protocol. An
explicitly fixed batch-1,024 representation could predict batch-one utility,
but it would be a cross-execution representation experiment: its matched
untouched baseline must use the same features, and native batch-one selector
scores remain a separate operational baseline. Record that distinction in
feature and label provenance.

The replication separates variation between discoveries, between base paths,
between anchor locations and between signed pairs. In a balanced nested design,
the variance of a mean selector-minus-uniform contrast is
`(variance_game + variance_base/B + variance_anchor/(B*A) + variance_pair/(B*A*P))/G`,
where G, B, A and P are the corresponding counts. Moment estimates can be noisy
or negative; that does not prove an absent variance component. Use the direct
variance of complete game-level contrasts for planning under the same design;
use the components chiefly to decide where extra samples would help. Resample
whole discovery vectors jointly because the contrasts reuse uniform measurements.
Incomplete observations require missing-data sensitivity instead of this complete
balanced-design calculation alone.

Keep uniform anchors and the natural discovery distribution. Report prespecified
clock-left and trajectory-phase strata descriptively; removing near-clock states
or equally weighting strata changes the target population. Independent games
do not become one dependent family merely because they start at the ordinary
initial board. Reused outcomes, shared sampled openings, cached confirmations
across games and augmented copies do create dependence or leakage; keep those
families together and report exact-state overlap separately.

Calibration, teacher training and held-out confirmation are separate stages.
Calibration data may be assigned to training only after the analysis plan is
fixed; it must never serve as held-out confirmation. This checkpoint's variance
also cannot automatically size an experiment on a different checkpoint or arm.
Choose the useful margin as an explicit engineering threshold: squared-bias
utility has no established conversion to learning gain or Elo. Do not infer
that threshold from favorable pilot effects. Fix the primary contrast, selection
rule, power alternative and confirmation count before observing fresh results;
report variance uncertainty and influential-game sensitivity. If conservative
confirmation is unaffordable, narrow the claim or retain an inconclusive result.

Extend the existing probe with explicit base replication and a future-only
endpoint under a new protocol version. Include every planned base, anchor and
pair in its denominator, including capped bases. Preserve protocol-3 artifacts
as immutable evidence and reject attempts to resume them under the new request.
CPU checks must cover independent streams, nested aggregation, missing-base
denominators, shared-selection cancellation and interrupted resume before GPU use.
Use one canonical nested base/anchor structure, with opening measurements only
when the requested endpoint includes them. A complete-case future mean requires
the full planned grid. Separate planned selector pair slots, known deduplicated
measurement slots, unavailable slots and actual launched/completed work; counting
only launched pairs conceals work prevented by a censored base or discovery.

The current pilot used 304 episode/root save units. Its 13.14 seconds outside
recorded search/play is not a serialization measurement. Larger representative
corpora require many more durable writes, and their serialization costs have
not been measured. The minimal next change is compact atomic JSON plus cumulative save
count, bytes and write time, retaining the durable save cadence. Compact encoding
of the existing pilot object uses 2,008,969 bytes instead of 3,479,777. Record that
persisted timing counters exclude their own snapshot write. Measure the new
one-game smoke stage before choosing a different storage format or fixing the
large-run resource ceiling.

The offline probe now reports `finite_sample_completion_range` alongside its
complete-case means and bootstrap intervals. For P planned pairs with observed
products summing to S and m explicitly missing products, the measurement range
is [(S-4m)/P, (S+4m)/P]. It averages all planned anchors and originating games.
A censored base leaves future utility observations in [-4,4]; a censored
discovery leaves method-minus-uniform differences in [-8,8]. A shared recorded
selection cancels exactly only after completed discovery. Malformed or
unfinished records are errors, rather than extra missing observations.
For example, [1,1,missing,missing] has range [-1.5,2.5]; incorrectly treating
missing signed products as nonnegative would manufacture a positive lower
bound of 0.5. These are deterministic completion ranges for the sampled grid,
not confidence intervals. Population missingness, Monte Carlo uncertainty,
head selection and repeated looks still require the prespecified confirmatory
analysis. The probe does not emit an adoption pass/fail flag.
The probe rejects nonfinite clock penalties or magnitudes above one before
creating progress or initializing CUDA inference. This keeps clock outcomes
and backed-up Q within the advertised support; the current penalty 0.05 is
unchanged and satisfies the bound. Out-of-support observed products are
errors rather than silently clipped values.
CPU evaluation of the two saved checkpoint-31 pilot artifacts reproduced all
12 existing complete-case comparison values to 1e-12. These fully observed
ranges collapse to the original values; this compatibility check adds no new
games or independent replication.

Predeclare a possible cheaper screen using the hierarchy: for method m and
uniform u, coarse lift lies between `U_state(m)-U_fine(u)` and
`U_fine(m)-U_state(u)`. Align discovery candidates and base paths, jointly
resample originating games, and include missing-data uncertainty. A lower
confidence bound above the useful margin establishes that particular coarse
comparison without collecting coarse labels; an upper bound below the margin
rules out useful gain. Otherwise collect direct coarse evidence. This screen
does not authorize repeatedly testing until a favorable result appears.

Compare methods on the same discovery games. Report effect sizes, intervals,
concentration, censoring and results across more than one frozen checkpoint.
If ranks do not outperform uniform on the independent target, the selection
failure is more directly established. If they do, the played-row symptom alone
does not justify replacing them. A wide interval is inconclusive.

Fit candidate heads on training-only data and test on separate originating
games/opening families; do not split adjacent rows from one trajectory across
training and validation. Cross two targets (current excess and independently
confirmed coarse future utility) with two objectives (current listwise and
direct regression). Match features, head capacity, initialization, training
data and tuning budgets; retain untouched rank and regret heads as baselines.
Relevant paper-based controls can then test whether its published mechanisms
transfer. Check recovery from poor initialization, calibration, held-out
selection lift and efficiency. Predeclare a common primary selection rule;
score scale and sampling concentration must not masquerade as better ordering.
A representation-frozen test isolates head learning but cannot establish the
behavior of joint trunk training. No such real-feature head fit is complete.

Source review identifies an exact feature interface: the current regret MLP
receives the shared 256-dimensional state encoding and has shape 256 to 128
to two outputs. Freeze the trunk and state encoder. A scoped pre-hook on a
separately loaded diagnostic model can capture the actual regret input without
copying `forward`. Hooks can change compilation behavior: ordinary outputs must
remain unchanged, and applying the original head to cached features must
reconstruct its recorded outputs exactly.
Record dtype, state identity and extraction/checkpoint/source settings. Failure
of that check blocks feature-cache use. Absolute ply is absent from the 46-plane
encoding despite influencing search temperature and the ply cap; identical
boards and capture-clock fields can therefore share features at different plies.
Keep this representation limitation identical across the four cells.

Use identical scalar 256-to-128-to-one readouts for the first offline 2-by-2.
This isolates rank learning from the joint head's auxiliary gradient; it does
not propose a production architecture replacement. Match examples, features,
family splits, initialization, minibatches and tuning budgets. Keep the untouched
joint head as a baseline. A separate interference check may retain both outputs
and both losses, changing only whether excess-loss gradients enter the shared
hidden layer; do not mix that change into the main four cells. Fixed replicate
aggregation is also required: unbiased signed labels do not make the nonlinear
listwise loss an unbiased objective on mean utility. Target-noise and objective
interactions remain part of the interpretation.

Predeclare the primary initialization as the checkpoint's trained rank head:
copy its shared hidden layer and rank output row into all four scalar cells
identically. This directly tests recovery from the observed trained ranking
state. Use the same minibatch seed and matched tuning budgets; do not center or
scale only one cell's initial scores. Three fixed random starts, seeds 0, 1 and
2, provide a separate learnability and optimization sensitivity. Do not choose
the best initialization after observing outcomes. Only the four primary
warm-start heads enter the budgeted fresh full-pool winner confirmations;
random-start sensitivities use the existing held-out sampled rows. Additional
winner confirmations would require a separately fixed sample allocation.

Training examples must include candidate occurrences sampled before observing
utility, with explicit per-game weights, rather than only the old selectors'
winners. Collect both target definitions at the same sampled starting states.
Extend the existing base-trajectory recorder with each step's selected Q, action
and mode so the current target can be reconstructed using terminal Q-domain Z
and mover signs. Saved `Window.ret` is generally a bootstrapped lambda-return;
it cannot replace terminal Z. Current probe trajectories contain only states,
so this telemetry extension is still needed. Calibration's selected states and
anchors alone are not a representative head-training corpus.

A residual teacher for the primary quantity should estimate `E[R given s,A,M]`.
If it uses additional inputs such as selected Q or root-tree features, integrate
over those inputs before squaring; squaring first silently changes the target
toward fine-context variability. Ordinary single-outcome residual observations
can train a teacher when their search/history protocol matches its endpoint;
expensive independent pairs are chiefly needed to validate bias and selection.
Do not mix retained-tree rollout labels with fresh-root confirmations without
testing or resolving that distribution difference.

One potentially economical candidate fits the action/mode residual teacher on
separate games, squares its detached predictions, integrates over on-policy
action/mode to obtain state utility, and only then averages along trajectories
and regresses one utility head onto that signal. A scalar residual teacher
would restore action cancellation. Using only the base path's sampled action
before suffix averaging is also a different target when actions affect game
length: an exact two-action example gives 0.5 versus the intended 0.375.
An independent action/mode draw conditional on each saved state can estimate
the integration; reusing the base action cannot. Its root-search cost must be
measured, or a state-level approximation independently validated.
This avoids multiplying a teacher
by the same terminal residual used in the random-length suffix, but the squared
teacher retains estimation and model-error bias. It is not an unbiased target;
independent paired anchors must validate it against direct signed pair-label
regression. No dense teacher pipeline has been implemented or validated.

A minimal teacher can condition on frozen state features, categorical action
and full/cheap mode, using residual MSE. Explicit absolute ply is a declared
teacher hypothesis, not an input to add only to a favored utility-head cell.
One fresh on-policy root and continuation per sampled teacher anchor gives
a training observation without rejection matching; using only accepted matched
pairs would select coverage when caps occur. Once the teacher is frozen,
`b_hat^2 - b_hat*(R1+R2) + R1*R2` from independent matched pairs estimates its
squared error against the conditional mean residual. Ordinary residual MSE
contains irreducible variation and cannot establish that squaring the teacher
gives useful rankings. Keep teacher-distilled labels distinct from the direct
paired-label regression cell.

Uniform anchors plus an independent action/mode root draw can supply future
teacher labels directly; a dense suffix table or second state-utility model
is unnecessary initially. Batched collection may later reduce costs, but both
members of a residual pair must have independent future streams, including
mode draws. Production's batch-shared mode draw would introduce covariance
if the members shared it. Use independent lane modes or separately seeded
waves, account for other shared randomness in clustering, and compare batched
against scalar collection before scientific use. No batched collector or
validated teacher/head feature cache has been implemented.

The completed checkpoint-31 pilots store four selected candidates per game,
not a training corpus of every alternative's features. The offline probe now
records full candidate pools for new discoveries, deduplicating states while
preserving ordered occurrences, sources, step/node locations and observed
rank/regret predictions. Original offer batches are retained to reproduce
floating-point reductions as well as selection ties. A compressed, checksummed
payload is cached once in the discovery episode; game summaries retain only
the four selections. Resume decodes and verifies the pool without inference.
This recording adds no random draws or model evaluations. Before judging a
newly fitted head, extract its frozen features from these full held-out pools,
let it choose, and obtain fresh confirmations for its choices. Full pools alone
are not independent utility labels or a completed head-fitting experiment.
Evaluating only the four old selectors' winners cannot establish the new head's
ranking.

A CPU synthetic storage benchmark used 21,636 occurrences, 13,156 distinct
varied states and 168 offer batches. Compact raw JSON was 4,122,838 bytes,
compressed data 1,006,656 bytes and its base64 form 1,342,208 bytes. The
single-pool artifact core was 1,359,051 bytes versus 24.79 MB for expanded
pretty JSON. Encoding took 0.075 seconds; decoding/validation took 0.320
seconds and full selection replay/validation 0.408 seconds. Cached snapshot
creation took a median 1.30 microseconds, and a pretty JSON write to memory
2.42 milliseconds. Progress still rewrites the cached bytes when saving each
episode or root attempt; this benchmark does not measure disk throughput,
large multi-game artifacts or production rollout overhead. No GPU inference
or retained benchmark files were used.

Retain the current heads if their independent future lift is useful and a
replacement does not improve robustly at comparable cost. Favor a replacement
only if it clears the prespecified margin against both current ranking and
uniform across independent games and multiple checkpoints, with acceptable
missing-data bounds. If reliable labels are not learnable by the tested heads,
investigate representation and conditioning rather than changing the loss again.

### E. Isolated experiments and production decision

Use isolated continuations from the same checkpoint with matched compute,
search settings and evaluation protocol. Separate target changes from objective
changes sufficiently to identify what helps. Version incompatible labels and
buffer statistics; specify their initialization instead of silently mixing old
and new quantities. Preserve an unmodified control arm. Execute GPU training
arms sequentially if only one GPU is available.

The arena's eval tool decides strength. Report equal-compute progress and
uncertainty, not just lower auxiliary loss or more favorable training samples.
Controlled continuation and arena evaluation are required before production
adoption; passing a diagnostic or head-fitting comparison alone is insufficient.
Reject changes that improve the diagnostic while degrading strength or making
collection impractically expensive. Do not combine the independent search
admission fix below into an RGSC causal comparison without controlling for it.
The authorized overnight resume retained the existing RGSC design and applied
the separately validated search correction. Keep subsequent experimental
checkpoints and configurations separate. Apply an RGSC replacement only after
the independent evidence and review gates above are met, with its label/buffer
transition specified. Report unresolved evidence without treating the resumed
run as an RGSC validation result.

## Immediate-win defect, correction and RGSC sensitivity

The original Python self-play search gives immediate wins (`win1`) and longer
exact wins (`win3`) the same positive-infinity admission score. When their tie
exceeds the candidate capacity, it can discard the immediate win. Final move
selection cannot choose an action absent from that shortlist. The tactic was
detected correctly and then lost during search selection.

The live configuration admits 16 candidates in both modes, even though cheap
search subsequently uses four. The omission occurs at that actual 16-slot
limit; it is not confined to tiny test budgets. In one saved position there
is one immediate win and 33 longer exact wins. The immediate win is omitted
and the game takes another 91 plies to finish.

The authorized correction uses one tactical ordering at admission, halving
and final selection: immediate wins, longer exact wins, then ordinary search
scores. Illegal, excluded and inactive slots remain excluded. Ordinary
prior/Gumbel expressions and the completed-Q and policy-target definitions
are unchanged. Tests cover capacities 1, 2, 4 and 16 in full and cheap modes,
hostile priors, actual live-capacity overflow and retention through halving.
Independent source review approved the correction; all 18 real CUDA search
cases passed, and the resumed trainer now uses the correction.
The Rust serving search already handles immediate wins before truncation.

Editing this Python file did not modify the original trainer. Search is
imported once, and CUDA graph recapture uses those loaded methods; there is no
reload path. The checkpoint-31 restart loaded the corrected source. The arena
exporter does not import this search.

### Measured occurrence

Two independent CPU implementations agree on the current-run totals. The
reusable measurement is `diagnose_control --audit`, under `search_selection`.
The engine independently verified 50 omitted positions, two from each current-run
iteration 0–24, plus 16 positions across the historical range: each omitted action
ends the game in a win for that row's mover, while the recorded action leaves
the game ongoing. Saved candidate/tactic/action fields are not rewritten by
outcome labeling.

| Saved windows | Immediate-win opportunities | Omitted from the 16 candidates | Fraction of opportunities |
|---|---:|---:|---:|
| g192p10, 0–24 | 77,542 | 60,966 | 78.62% |
| g160, 51–110, before ranking reversal | 190,777 | 143,739 | 75.34% |
| g160, 111 | 2,649 | 1,974 | 74.52% |
| g160, 112–140, after reversal | 89,654 | 72,933 | 81.35% |

These are dependent opportunities, often repeated in the same game. In the
current-run 0–24 audit they constitute 1.24% of 4,915,200 recorded positions. Full
search omitted 31,185 of 39,703 opportunities; cheap search omitted 29,781 of
37,839. Every recorded non-immediate choice at such a position coincided with
the immediate win being absent from the candidate array.

Reconstructing complete games in that audit gives 16,576 games with an immediate
winning opportunity; 5,586 (33.70% of that subset) deferred the first one. Those
games took a mean 22.71 plies from that opportunity to the observed end, median
17, 90th percentile 51, maximum 167; immediate conversion would take one ply.
Cutting at the first opportunity removes 121,248 rows across those complete
games. This is not a measured speedup: fixed-step collection can instead play
more games, and position-dependent search cost can change.

All 60,966 omitted rows in the current-run 0–24 audit have known eventual winning outcomes.
Historical known outcomes also all won, with some unfinished labels remaining
at the audit snapshot. Demonstrated harm is repeated delay and unnecessary
play, not demonstrated lost games or a measured strength deficit. The defect
was already common before g160's ranking reversal, so it cannot uniquely
explain the onset of that reversal.

### Does correcting those trajectories remove the RGSC symptom?

For a deliberately limited sensitivity check, end each saved complete game
at its first detected immediate win, set that final selected Q and mover's
terminal return to one, retain earlier logged predictions, and recompute the
signed suffix labels. Compare original and corrected labels on the same
retained prefix using renormalized saved ranks. Games with no immediate win
retain their original trajectory. This prevents removed rows alone from
masquerading as a label improvement.

The current-run corrected-prefix lift remains negative in every iteration
2–24, with all 23 game-bootstrap intervals below zero. At 24 the original
same-prefix lift is -0.01492; after relabeling it is -0.01715, interval
[-0.02328, -0.01126]. An independent reviewer checked direct algebra, mover
parity, multiple environments, actor/window boundaries and extreme discarded
scores. No recorded first-win mover had a conflicting eventual result.

The same sensitivity preserves g160's transition: corrected-prefix lift is
positive throughout 51–110, near zero at 111, and negative throughout 112–140,
with all 29 late intervals below zero. At 140 it is -0.01114, interval
[-0.01380, -0.00865].

This is a sensitivity calculation with frozen predictions, not the exact
distribution generated by the new search. Corrected search can change earlier
actions, tree reuse, outcomes and later training. Nevertheless, the saved
ranking deficit does not disappear merely by removing these delayed endings.
Run the frozen-checkpoint comparison under corrected search before deciding
which RGSC target and head change is justified; use that same corrected search
in every subsequent comparison arm.
