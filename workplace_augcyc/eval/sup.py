import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
"""
how performance varies as the proportion of can-pair b in train varies
"""

sns.set_theme()
sns.set(font_scale=1.4)
sns.set_style("whitegrid")
baseline = 0.0336857286421176
only_unpaired = 0.020579816522603562

data = """cv-0.1/output.txt:prior: 0.0301580595258377
cv-0.1/output.txt:prior: 0.023184082250555255
cv-0.1/output.txt:prior: 0.03149158871415021
cv-0.1/output.txt:prior: 0.022024167867715775
cv-0.1/output.txt:prior: 0.027796934917017638
cv-0.1/output.txt:prior: 0.02901836952807701
cv-0.1/output.txt:prior: 0.025437896072962683
cv-0.1/output.txt:prior: 0.035203584206347685
cv-0.1/output.txt:prior: 0.026169588118646225
cv-0.1/output.txt:prior: 0.029341022443926296
cv0.1/output.txt:prior: 0.017975663984889395
cv0.1/output.txt:prior: 0.019695314412241975
cv0.1/output.txt:prior: 0.020190524706109235
cv0.1/output.txt:prior: 0.018510488477971678
cv0.1/output.txt:prior: 0.02118242480606958
cv0.1/output.txt:prior: 0.017926918038818403
cv0.1/output.txt:prior: 0.017544978463527205
cv0.1/output.txt:prior: 0.01883316284887758
cv0.1/output.txt:prior: 0.017825109289578582
cv0.1/output.txt:prior: 0.01900823106526198
cv-0.2/output.txt:prior: 0.026441342035433305
cv-0.2/output.txt:prior: 0.024903635719410122
cv-0.2/output.txt:prior: 0.021180766516272796
cv-0.2/output.txt:prior: 0.02323179529797132
cv-0.2/output.txt:prior: 0.024745565529769153
cv-0.2/output.txt:prior: 0.025038926396537754
cv-0.2/output.txt:prior: 0.023895936485285794
cv-0.2/output.txt:prior: 0.021321670471825014
cv-0.2/output.txt:prior: 0.027425768573289065
cv-0.2/output.txt:prior: 0.029330128109925326
cv0.2/output.txt:prior: 0.020043211106513397
cv0.2/output.txt:prior: 0.016895628757372147
cv0.2/output.txt:prior: 0.017119728533308332
cv0.2/output.txt:prior: 0.017586192644891994
cv0.2/output.txt:prior: 0.01800127286553199
cv0.2/output.txt:prior: 0.01796961942152509
cv0.2/output.txt:prior: 0.020609357859623593
cv0.2/output.txt:prior: 0.018815746075318503
cv0.2/output.txt:prior: 0.016962686681714367
cv0.2/output.txt:prior: 0.019572893365971525
cv-0.3/output.txt:prior: 0.025504989950628724
cv-0.3/output.txt:prior: 0.023047169489236095
cv-0.3/output.txt:prior: 0.023034618797108625
cv-0.3/output.txt:prior: 0.021445806324181863
cv-0.3/output.txt:prior: 0.019550713169771895
cv-0.3/output.txt:prior: 0.02124266470849206
cv-0.3/output.txt:prior: 0.023702176141317338
cv-0.3/output.txt:prior: 0.027082706434567606
cv-0.3/output.txt:prior: 0.017590741910496338
cv-0.3/output.txt:prior: 0.021704227304303183
cv0.3/output.txt:prior: 0.015480229095235606
cv0.3/output.txt:prior: 0.016195158337667444
cv0.3/output.txt:prior: 0.017293750649539308
cv0.3/output.txt:prior: 0.017804302763455657
cv0.3/output.txt:prior: 0.017476596925015794
cv0.3/output.txt:prior: 0.017415131055274535
cv0.3/output.txt:prior: 0.016750341118741442
cv0.3/output.txt:prior: 0.016543923294591454
cv0.3/output.txt:prior: 0.018890296278878473
cv0.3/output.txt:prior: 0.015587637100883036
cv-0.4/output.txt:prior: 0.02436504356104653
cv-0.4/output.txt:prior: 0.018768497348697788
cv-0.4/output.txt:prior: 0.023758165373812798
cv-0.4/output.txt:prior: 0.01906879833457011
cv-0.4/output.txt:prior: 0.024388375654348274
cv-0.4/output.txt:prior: 0.02301997365571442
cv-0.4/output.txt:prior: 0.021840405866022452
cv-0.4/output.txt:prior: 0.01957516946186484
cv-0.4/output.txt:prior: 0.018036853895911784
cv-0.4/output.txt:prior: 0.019095476903771485
cv0.4/output.txt:prior: 0.017183963368028764
cv0.4/output.txt:prior: 0.017189641299499486
cv0.4/output.txt:prior: 0.01543566523653921
cv0.4/output.txt:prior: 0.016380466906049055
cv0.4/output.txt:prior: 0.01812747370260901
cv0.4/output.txt:prior: 0.017133317705718187
cv0.4/output.txt:prior: 0.015926156677087265
cv0.4/output.txt:prior: 0.015046043388886604
cv0.4/output.txt:prior: 0.014914110161453728
cv0.4/output.txt:prior: 0.015519502908254412
cv-0.5/output.txt:prior: 0.01753355374827149
cv-0.5/output.txt:prior: 0.017641343699975144
cv-0.5/output.txt:prior: 0.019224407612892975
cv-0.5/output.txt:prior: 0.02208567687302218
cv-0.5/output.txt:prior: 0.01695198869951061
cv-0.5/output.txt:prior: 0.021691880364536887
cv-0.5/output.txt:prior: 0.019996799408606176
cv-0.5/output.txt:prior: 0.02037727009103792
cv-0.5/output.txt:prior: 0.017948964185040734
cv-0.5/output.txt:prior: 0.018826879465427417
cv0.5/output.txt:prior: 0.0148343826157317
cv0.5/output.txt:prior: 0.015528821748331399
cv0.5/output.txt:prior: 0.015241371035511014
cv0.5/output.txt:prior: 0.016104958137795968
cv0.5/output.txt:prior: 0.016442769082813256
cv0.5/output.txt:prior: 0.016697084135754662
cv0.5/output.txt:prior: 0.01691633080591487
cv0.5/output.txt:prior: 0.016541418001217407
cv0.5/output.txt:prior: 0.015500561382720733
cv0.5/output.txt:prior: 0.014128593753054423
cv-0.6/output.txt:prior: 0.019590050057228803
cv-0.6/output.txt:prior: 0.019232875126248734
cv-0.6/output.txt:prior: 0.02292341519478933
cv-0.6/output.txt:prior: 0.017016091299649923
cv-0.6/output.txt:prior: 0.01681826420566101
cv-0.6/output.txt:prior: 0.019401098221684756
cv-0.6/output.txt:prior: 0.019339150484642246
cv-0.6/output.txt:prior: 0.019989267963130348
cv-0.6/output.txt:prior: 0.020619761443531535
cv-0.6/output.txt:prior: 0.018228796446630786
cv0.6/output.txt:prior: 0.01676261280472722
cv0.6/output.txt:prior: 0.014630459259420896
cv0.6/output.txt:prior: 0.016413842347088602
cv0.6/output.txt:prior: 0.015459026198916275
cv0.6/output.txt:prior: 0.0146645939584598
cv0.6/output.txt:prior: 0.015319114300321644
cv0.6/output.txt:prior: 0.014397012915329497
cv0.6/output.txt:prior: 0.016480087324673855
cv0.6/output.txt:prior: 0.014621965057844041
cv0.6/output.txt:prior: 0.013588971892106836
cv-0.7/output.txt:prior: 0.01454271113586177
cv-0.7/output.txt:prior: 0.017243078421682646
cv-0.7/output.txt:prior: 0.016670212813385824
cv-0.7/output.txt:prior: 0.02105104048661374
cv-0.7/output.txt:prior: 0.021961247527465776
cv-0.7/output.txt:prior: 0.01511219502409321
cv-0.7/output.txt:prior: 0.018193880431121523
cv-0.7/output.txt:prior: 0.01988318196839096
cv-0.7/output.txt:prior: 0.016364327153044568
cv-0.7/output.txt:prior: 0.0164869956821291
cv0.7/output.txt:prior: 0.01490830764775171
cv0.7/output.txt:prior: 0.014067922473205841
cv0.7/output.txt:prior: 0.015177563502158455
cv0.7/output.txt:prior: 0.01591656133590795
cv0.7/output.txt:prior: 0.012639420599756297
cv0.7/output.txt:prior: 0.015951825115321713
cv0.7/output.txt:prior: 0.01664762799673278
cv0.7/output.txt:prior: 0.01627615244047274
cv0.7/output.txt:prior: 0.014638170208612189
cv0.7/output.txt:prior: 0.01630872356645389"""


def parse_data(data, unpaired=True):
    cvdata = {}
    for l in data.split("\n"):
        s, v = l.split("prior:")
        v = float(v)
        s = s.split("/")[0]
        s = float(s.replace("cv", ""))

        if unpaired:
            include = s>0
        else:
            include = s<0

        if include:
            if abs(s) not in cvdata:
                cvdata[abs(s)] = [v]
            else:
                cvdata[abs(s)].append(v)
    print(cvdata)
    pltdata = [(k, np.mean(v), np.std(v), np.std(v),) for k, v in cvdata.items()]
    pltdata = sorted(pltdata, key=lambda x: x[0])
    return pltdata

def plotcv(data, unpaired=True, fmt="ro:", label="with unpaired"):

    pltdata = parse_data(data, unpaired=unpaired)
    x = [xye[0] for xye in pltdata]
    y = [xye[1] for xye in pltdata]
    dylower = [xye[2] for xye in pltdata]
    dyupper = [xye[3] for xye in pltdata]
    # plt.plot(x, y, "ro:", label="with unpaired")
    markers, caps, bars = plt.errorbar(x, y, yerr=[dylower, dyupper], fmt=fmt, label=label, capthick=1.0, capsize=2, elinewidth=1.5)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    return x, y, dylower, dyupper

x, y, _, _ = plotcv(data, unpaired=True, fmt="ro:", label="with unpaired")
x, y, _, _ = plotcv(data, unpaired=False, fmt="bx--", label="without unpaired")

xmin = min(x) - 0.05
xmax = max(x) + 0.05

plt.grid(axis="x")

plt.hlines(y=baseline, xmin=xmin, xmax=xmax, colors="k", linestyles="-", label="identity baseline", lw=2)
plt.hlines(y=only_unpaired, xmin=xmin, xmax=xmax, colors="k", linestyles="-.", label="only unpaired", lw=2)
plt.xlim([xmin, xmax])
plt.ylabel(r"mean $\rm{\Delta_{\rm{sample}} (C, C')}$")
plt.xlabel("Proportion of can-pair B used in training")
plt.legend(bbox_to_anchor=[0.5, 1.12], loc='center', ncol=2)
plt.savefig("sup.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"} )
plt.savefig("sup.eps", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"} )
