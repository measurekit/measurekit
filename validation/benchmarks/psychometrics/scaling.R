# Scale scores and standard setting have no reference of comparable computation: mirt reports a scale score for a response
# pattern, not for each summed score. The tasks are measured for measurekit alone.

bench_task("scaling.conversionTableEapSum", "A conversion table of the posterior mean of theta given each summed score", "40 items")
bench_task("scaling.conversionTableTrueScore", "A conversion table of the theta of each summed true score", "40 items")
bench_task("standardSetting.bookmarkCutScore", "The cut score of a bookmark in an ordered item booklet", "40 items")
bench_task("standardSetting.classificationAccuracy", "Accuracy and consistency of the classification by two cuts", "2,000 persons")
bench_task("interop.toMirtSlopeIntercept", "Conversion of 2PL items to mirt's slope-intercept form", "10,000 items")
bench_task("interop.toMirtIrt", "Conversion of 2PL items to mirt's IRT form", "10,000 items")
