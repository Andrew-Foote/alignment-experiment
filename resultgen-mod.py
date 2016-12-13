import itertools
import enum
import random
import statistics
from matplotlib import pyplot
import os
import pickle

def common_value(iterable):
    iterable = tuple(iterable)
    x = iterable[0]
    for y in iterable:
        if x != y:
            raise ValueError('no common value')
    return x


def linedesc(data):
    
    try:
        mode = '{:5.1f}'.format(statistics.mode(data))
    except statistics.StatisticsError:
        mode = '  N/A'
    
    return ' '.join((
        '{mean:6.3f}', '{medianl:11.3f}', '{medianh:11.3f}', mode,
        '{sd:6.3f}', '{minm:6.3f}', '{maxm:6.3f}'
    )).format(
        mean = statistics.mean(data),
        medianl = statistics.median_low(data),
        medianh = statistics.median_high(data),
        sd = statistics.stdev(data),
        minm = min(data),
        maxm = max(data)
    )


class VerbType(enum.Enum):
    
    UERG = 'uerg' # unergative (intransitive with agent-like argument)
    UACC = 'uacc' # unaccusative (intransitive with patient-like argument)
    MT = 'mt' # (mono)transitive

    def it(self): # return true iff intransitive
        return self == VerbType.UERG or self == VerbType.UACC


class Condition(enum.Enum):
    
    ZC_A = 'A Zero-Coded' # zero-coded A, overtly coded P
    ZC_P = 'P Zero-Coded' # overtly coded A, zero-coded P
    BOTH_OC = 'Both Overtly Coded' # both A and P overtly coded

    def __str__(self):
        return self.value


class Argument(enum.Enum):
    
    ZC = '-'
    OC_A = '-low'
    OC_P = '-ray'

    @classmethod
    def a(cls, condition):
        if condition == Condition.ZC_A:
            return Argument.ZC
        else:
            return Argument.OC_A

    @classmethod
    def p(cls, condition):
        if condition == Condition.ZC_P:
            return Argument.ZC
        else:
            return Argument.OC_P

    def overt(self):
        return self == Argument.OC_A or self == Argument.OC_P

    def __str__(self):
        return self.value

class Alignment(enum.Enum):

    ACC = 'acc'
    ERG = 'erg'
    TRI = 'tri'
    

class Trial:
    
    def __init__(self, verb_type, args):
        self.verb_type = verb_type
        self.args = args

    def __repr__(self):
        return "Trial(verb_type=" + repr(self.verb_type) + ", args=" + repr(self.args) + ")"

    def __str__(self):
        return "V[+" + self.verb_type.value + "]" + "".join(" " + arg.value for arg in self.args)


class MTTrial(Trial):
    
    def complete(self):
        return len(self.args) == 2

    def accurate(self, condition):
        return self.args == (Argument.a(condition), Argument.p(condition))


class ITTrial(Trial):

    def complete(self):
        return len(self.args) == 1

    def alignment(self, condition):
                
        if self.args[0] == Argument.a(condition):
            return Alignment.ACC
        elif self.args[0] == Argument.p(condition):
            return Alignment.ERG
        else:
            return Alignment.TRI

    def overt_s(self):
        
        return self.args[0].overt()

class IndividualDataset:
    
    @classmethod
    def generate(cls, condition, n_mt, n_uerg, n_uacc):

        # Each of the constants below is a tuple representing the two
        # parameters of a beta distribution from which a probability of some
        # event (e.g. the recollection of a particular noun word) is drawn
        # during dataset generation. A beta-distribution with parameters
        # (a, b) has minimum 0, maximum 1, mean
        #
        #   mu = a/(a + b)
        #
        # and s. d.
        #
        #   sigma = sqrt(ab)/((a + b) sqrt(a + b + 1))
        #         = sqrt(mu(1 - mu)/(a + b + 1)).
        #
        # Qualitatively, the mean is between 0 and 1, and is close to 0 if
        # a/b is small and is close to 1 if a/b is large. The variance is
        # larger if mu is close to 1/2 and smaller if mu is close to 0 or 1,
        # all else being equal; however, the variance is also large if the
        # sum a + b (which is independent of mu) is large, and for values of
        # a + b > 1 the effect of this dwarfs that of mu's closeness to 1/2
        # (since mu(1 - mu) < 1).

        # determines probability of noun recollection
        N_RECALL_PARAMS = (64/3, 1/3)

        # determines probability of transitive case-coding system
        # recollection (mean is slightly closer to 1 in condition 3 than
        # in other conditions; first item is params for conditions 1-2,
        # second item is params for condition 3)
        MT_SYS_RECALL_PARAMS = ((14/40, 1/40), (19/40, 1/40))

        # determines probability of A-coding suffix (-low) recollection
        A_SUF_RECALL_PARAMS = (54/80, 1/80)

        # determines probability of P-coding suffix (-ray) recollection
        P_SUF_RECALL_PARAMS = (54/80, 1/80)

        # determines probability of extension of transitive case-coding
        # system (*if this has already been recollected*) to intransitive
        # sentences (mean is slightly closer to 1 in condition 3 than
        # in other conditions; first item is params for conditions 1-2,
        # second item is params for condition 3)
        IT_SYS_RECALL_PARAMS = ((11/36, 1/36), (14/36, 1/36))

        # determines probability of zero-coding of S being opted for due to
        # efficiency considerations (mean is close to 1 in conditions 1-2
        # but close to 0 in condition 3---this is the major difference
        # between conditions; first item is params for conditions 1-2,
        # second item is parmas for condition 3)
        ZERO_CODE_PARAMS = ((9/45, 1/45), (1/42, 14/42))

        # determines probability of zero-coding being overridden, if it was
        # already opted for, due to verbal unergativity and A-coding suffix
        # being added
        OVERT_UNERG_PARAMS = (3/12, 4/12)

        # determines probability of zero-coding being overridden, if it was
        # already opted for, due to verbal unaccusativity and P-coding suffix
        # being added
        OVERT_UNACC_PARAMS = (3/12, 2/12)

        # determines probability of accusative alignment transferring over
        # from English, *if zero-coding was not already opted for*;
        # different for unergative and unaccusative verbs
        ACC_ALIGN_UERG_PARAMS = (64/64, 1/64)
        ACC_ALIGN_UACC_PARAMS = (6/18, 1/18)
    
        p_n_recall = random.betavariate(*N_RECALL_PARAMS)
        p_mt_sys_recall = random.betavariate(*MT_SYS_RECALL_PARAMS[condition == Condition.BOTH_OC])
        p_a_suf_recall = (random.betavariate(*A_SUF_RECALL_PARAMS), 0)[condition == Condition.ZC_A]
        p_p_suf_recall = (random.betavariate(*P_SUF_RECALL_PARAMS), 0)[condition == Condition.ZC_P]
        p_it_sys_recall = random.betavariate(*IT_SYS_RECALL_PARAMS[condition == Condition.BOTH_OC])
        p_zero_code = random.betavariate(*ZERO_CODE_PARAMS[condition == Condition.BOTH_OC])
        p_overt_uerg = random.betavariate(*OVERT_UNERG_PARAMS)
        p_overt_uacc = random.betavariate(*OVERT_UNACC_PARAMS)
        p_acc_align_uerg = random.betavariate(*ACC_ALIGN_UERG_PARAMS)
        p_acc_align_uacc = random.betavariate(*ACC_ALIGN_UACC_PARAMS)

        trials = []

        for mt_trial in range(n_mt):
            args = []
            mt_sys_recall = random.random() < p_mt_sys_recall
            if random.random() < p_n_recall:
                args.append((Argument.ZC, Argument.a(condition))[mt_sys_recall and random.random() < p_a_suf_recall])
            if random.random() < p_n_recall:
                args.append((Argument.ZC, Argument.p(condition))[mt_sys_recall and random.random() < p_p_suf_recall])
            trials.append(MTTrial(VerbType.MT, tuple(args)))

        for uerg_trial in range(n_uerg):
            if random.random() < p_n_recall:
                if random.random() < p_mt_sys_recall and random.random() < p_it_sys_recall:
                    if random.random() < p_zero_code:
                        if random.random() < p_overt_uerg:
                            arg = (Argument.ZC, Argument.a(condition))[random.random() < p_a_suf_recall]
                        else:
                            arg = Argument.ZC
                    else:
                        if random.random() < p_acc_align_uerg:
                            arg = (Argument.ZC, Argument.a(condition))[random.random() < p_a_suf_recall]
                        else:
                            arg = (Argument.ZC, Argument.p(condition))[random.random() < p_p_suf_recall]
                else:
                    arg = Argument.ZC
                trials.append(ITTrial(VerbType.UERG, (arg,)))
            else:
                trials.append(ITTrial(VerbType.UERG, ()))

        for uacc_trial in range(n_uacc):
            if random.random() < p_n_recall:
                if random.random() < p_mt_sys_recall and random.random() < p_it_sys_recall:
                    if random.random() < p_zero_code:
                        if random.random() < p_overt_uacc:
                            arg = (Argument.ZC, Argument.p(condition))[random.random() < p_p_suf_recall]
                        else:
                            arg = Argument.ZC
                    else:
                        if random.random() < p_acc_align_uerg:
                            arg = (Argument.ZC, Argument.a(condition))[random.random() < p_a_suf_recall]
                        else:
                            arg = (Argument.ZC, Argument.p(condition))[random.random() < p_p_suf_recall]
                else:
                    arg = Argument.ZC
                trials.append(ITTrial(VerbType.UACC, (arg,)))
            else:
                trials.append(ITTrial(VerbType.UACC, ()))

        return cls(condition, trials)

    def __init__(self, condition, trials):
        self.condition = condition
        self.trials = tuple(trials)

    def size(self):
        return len(self.trials)

    def filter(self, f):
        return IndividualDataset(self.condition, filter(f, self.trials))

    def mt_part(self):
        return self.filter(lambda trial: trial.verb_type == VerbType.MT)

    def it_part(self):
        return self.filter(lambda trial: isinstance(trial, ITTrial))

    def uerg_part(self):
        return self.filter(lambda trial: trial.verb_type == VerbType.UERG)

    def uacc_part(self):
        return self.filter(lambda trial: trial.verb_type == VerbType.UACC)

    def complete_part(self):
        return self.filter(lambda trial: trial.complete())
    
    def accurate_part(self):
        return self.mt_part().filter(lambda trial: trial.accurate(self.condition))
    
    def accuracy_rate(self):
        return self.accurate_part().size()/self.mt_part().complete_part().size()

    def accusative_part(self):
        return self.it_part().complete_part().filter(lambda trial: trial.alignment(self.condition) == Alignment.ACC)

    def accusativity_rate(self):
        return self.accusative_part().size()/self.it_part().complete_part().size()

    def ergative_part(self):
        return self.it_part().complete_part().filter(lambda trial: trial.alignment(self.condition) == Alignment.ERG)

    def ergativity_rate(self):
        return self.ergative_part().size()/self.it_part().complete_part().size()

    def overt_s_part(self):
        return self.it_part().complete_part().filter(lambda trial: trial.overt_s())

    def overt_s_rate(self):
        return self.overt_s_part().size()/self.it_part().complete_part().size()

    def describe(self):
        mtp = self.mt_part()
        complete_mtp = mtp.complete_part()
        print('Of the', mtp.size(), 'transitive sentences,', complete_mtp.size(), 'were complete.')
        print('Of those,', self.accurate_part().size(), 'were accurate.')
        print()
        itp = self.it_part()
        complete_itp = itp.complete_part()
        print('Of the', itp.size(), 'intransitive sentences,', complete_itp.size(), 'were complete.')
        print(
            'Of those,', complete_itp.ergative_part().size(), 'were consistent with ergative alignment',
            'and', complete_itp.accusative_part().size(), 'were consistent with accusative alignment.'
        )
        print()
        uergp = self.uerg_part()
        complete_uergp = uergp.complete_part()
        print('Of the', uergp.size(), 'unergative sentences,', complete_uergp.size(), 'were complete.')
        print(
            'Of those,', complete_uergp.ergative_part().size(), 'were consistent with ergative alignment',
            'and', complete_uergp.accusative_part().size(), 'were consistent with accusative alignment.'
        )
        print()
        uaccp = self.uacc_part()
        complete_uaccp = uaccp.complete_part()
        print('Of the', uaccp.size(), 'unaccusative sentences,', complete_uaccp.size(), 'were complete.')
        print(
            'Of those,', complete_uaccp.ergative_part().size(), 'were consistent with ergative alignment',
            'and', complete_uaccp.accusative_part().size(), 'were consistent with accusative alignment.'
        )
        

class GeneralDataset:

    def __init__(self, idata):
        self.idata = tuple(idata)
        self.condition = common_value(
            dataset.condition for dataset in self.idata
        )

    def size(self):
        return len(self.idata)

    def condition(self):
        return self.idata[0].condition

    def mt_slice(self):
        return GeneralDataset(dataset.mt_part() for dataset in self.idata)

    def it_slice(self):
        return GeneralDataset(dataset.it_part() for dataset in self.idata)

    def uerg_slice(self):
        return GeneralDataset(dataset.uerg_part() for dataset in self.idata)

    def uacc_slice(self):
        return GeneralDataset(dataset.uacc_part() for dataset in self.idata)

    def n_trials_data(self):
        return tuple(dataset.size() for dataset in self.idata)
    
    def complete_slice(self):
        return GeneralDataset(dataset.complete_part() for dataset in self.idata)

    def accurate_slice(self):
        return GeneralDataset(dataset.accurate_part() for dataset in self.idata)

    def accuracy_rate_data(self):
        return tuple(dataset.accuracy_rate() for dataset in self.idata)

    def accurate_part(self, threshold):
        return GeneralDataset(
            filter(
                (lambda dataset: dataset.accuracy_rate() > threshold),
                self.idata
            )
        )

    def accusative_slice(self):
        return GeneralDataset(dataset.accusative_part() for dataset in self.idata)

    def accusativity_rate_data(self):
        return tuple(dataset.accusativity_rate() for dataset in self.idata)
    
    def ergative_slice(self):
        return GeneralDataset(dataset.ergative_part() for dataset in self.idata)

    def ergativity_rate_data(self):
        return tuple(dataset.ergativity_rate() for dataset in self.idata)

    def overt_s_slice(self):
        return GeneralDataset(dataset.overt_s_part() for dataset in self.idata)

    def overt_s_rate_data(self):
        return tuple(dataset.overt_s_rate() for dataset in self.idata)

    def describe(self, threshold):
        r = self.accurate_part(threshold).it_slice()

        return """\
Condition: {c}

Monotransitive sentences
                            Mean Median (h.) Median (l.)  Mode  S. d.    Min    Max
  N complete              {nmtcompl}
  N accurate              {naccurate}
  N accurate / N complete {paccurate}

There are {nincl} participants with N accurate / N complete > {threshold}. The
following statistics are measured over these participants only.

Intransitive sentences (of either type)
                              Mean Median (h.) Median (l.)  Mode  S. d.    Min    Max
  N complete                {nitcompl}
  N accusative              {nitacc}
  N accusative / N complete {pitacc}
  N ergative                {niterg}
  N ergative / N complete   {piterg}
  N overt S                 {nitos}
  N overt S / N complete    {pitos}

Intransitive sentences (unergative)
                              Mean Median (h.) Median (l.)  Mode  S. d.    Min    Max
  N complete                {nuergcompl}
  N accusative              {nuergacc}
  N accusative / N complete {puergacc}
  N ergative                {nuergerg}
  N ergative / N complete   {puergerg}
  N overt S                 {nuergos}
  N overt S / N complete    {puergos}

Intransitive sentences (unaccusative)
                              Mean Median (h.) Median (l.)  Mode  S. d.    Min    Max
  N complete                {nuacccompl}
  N accusative              {nuaccacc}
  N accusative / N complete {puaccacc}
  N ergative                {nuaccerg}
  N ergative / N complete   {puaccerg}
  N overt S                 {nuaccos}
  N overt S / N complete    {puaccos}

""".format(
    c = self.condition.value,
    nmtcompl = linedesc(self.mt_slice().complete_slice().n_trials_data()),
    naccurate = linedesc(self.accurate_slice().n_trials_data()),
    paccurate = linedesc(self.accuracy_rate_data()),
    nincl = r.size(),
    threshold = threshold,
    nitcompl = linedesc(r.complete_slice().n_trials_data()),
    nitacc = linedesc(r.accusative_slice().n_trials_data()),
    pitacc = linedesc(r.accusativity_rate_data()),
    niterg = linedesc(r.ergative_slice().n_trials_data()),
    piterg = linedesc(r.ergativity_rate_data()),
    nitos = linedesc(r.overt_s_slice().n_trials_data()),
    pitos = linedesc(r.overt_s_rate_data()),
    nuergcompl = linedesc(r.uerg_slice().complete_slice().n_trials_data()),
    nuergacc = linedesc(r.uerg_slice().accusative_slice().n_trials_data()),
    puergacc = linedesc(r.uerg_slice().accusativity_rate_data()),
    nuergerg = linedesc(r.uerg_slice().ergative_slice().n_trials_data()),
    puergerg = linedesc(r.uerg_slice().ergativity_rate_data()),
    nuergos = linedesc(r.uerg_slice().overt_s_slice().n_trials_data()),
    puergos = linedesc(r.uerg_slice().overt_s_rate_data()),
    nuacccompl = linedesc(r.uacc_slice().complete_slice().n_trials_data()),
    nuaccacc = linedesc(r.uacc_slice().accusative_slice().n_trials_data()),
    puaccacc = linedesc(r.uacc_slice().accusativity_rate_data()),
    nuaccerg = linedesc(r.uacc_slice().ergative_slice().n_trials_data()),
    puaccerg = linedesc(r.uacc_slice().ergativity_rate_data()),
    nuaccos = linedesc(r.uacc_slice().overt_s_slice().n_trials_data()),
    puaccos = linedesc(r.uacc_slice().overt_s_rate_data())
)


def go(conditions = (Condition.ZC_P, Condition.BOTH_OC), n_mt = 20, n_uerg = 10, n_uacc = 10, threshold = 0.9):

    data = []
    
    if Condition.ZC_A in conditions:
        print("-----------------------------------------------------------------")
        subdata = GeneralDataset(IndividualDataset.generate(Condition.ZC_A, n_mt, n_uerg, n_uacc) for i in range(25))
        print(subdata.describe(threshold))
        data.append(subdata)

    if Condition.ZC_P in conditions:
        print("-----------------------------------------------------------------")
        subdata = GeneralDataset(IndividualDataset.generate(Condition.ZC_P, n_mt, n_uerg, n_uacc) for i in range(25))
        print(subdata.describe(threshold))
        data.append(subdata)

    if Condition.BOTH_OC in conditions:
        print("-----------------------------------------------------------------")
        subdata = GeneralDataset(IndividualDataset.generate(Condition.BOTH_OC, n_mt, n_uerg, n_uacc) for i in range(25))
        print(subdata.describe(threshold))
        data.append(subdata)

    del subdata

    fig1 = pyplot.figure(1, figsize=(5, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_xticks(tuple(range(len(conditions))))
    ax1.set_xticklabels(tuple(i for i in range(len(conditions))))
    ax1.set_yticks(tuple(y/10 for y in range(11)))
    ax1.scatter(
        tuple(itertools.chain.from_iterable(
            (i,) * subdata.accurate_part(threshold).size() for i,subdata in enumerate(data)
        )),
        tuple(itertools.chain.from_iterable(
            subdata.accurate_part(threshold).overt_s_rate_data() for subdata in data
        )),
        s = 500, alpha = 0.25, c = (0, 0, 0), edgecolors = 'none'
    )
    ax1.set_title('Coding of intransitive sentences at test')
    ax1.set_xlim((-0.5, len(conditions) - 0.5))
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Mean proportion with overtly-coded argument')

    fig2 = pyplot.figure(2, figsize=(6, 6))    
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xticks(tuple(range(len(conditions))))
    ax2.set_xticklabels(tuple(i for i in range(len(conditions))))
    ax2.set_yticks(tuple(y/10 for y in range(11)))
    ax2.scatter(
        tuple(itertools.chain.from_iterable(
            (i,) * subdata.accurate_part(threshold).uerg_slice().size() for i,subdata in enumerate(data)
        )),
        tuple(itertools.chain.from_iterable(
            subdata.accurate_part(threshold).uerg_slice().overt_s_rate_data() for subdata in data
        )),
        s = 500, alpha = 0.25, c = (0, 0, 0), edgecolors = 'none'
    )
    ax2.set_xlim((-0.5, len(conditions) - 0.5))
    ax2.set_title('Coding of intransitive sentences\nwith volitional verbs at test')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Mean proportion with overtly-coded argument')

    fig3 = pyplot.figure(3, figsize=(6, 6))
    ax2 = fig3.add_subplot(1, 1, 1)
    ax2.set_xticks(tuple(range(len(conditions))))
    ax2.set_xticklabels(tuple(i for i in range(len(conditions))))
    ax2.set_yticks(tuple(y/10 for y in range(11)))
    ax2.scatter(
        tuple(itertools.chain.from_iterable(
            (i,) * subdata.accurate_part(threshold).uacc_slice().size() for i,subdata in enumerate(data)
        )),
        tuple(itertools.chain.from_iterable(
            subdata.accurate_part(threshold).uacc_slice().overt_s_rate_data() for subdata in data
        )),
        s = 500, alpha = 0.25, c = (0, 0, 0), edgecolors = 'none'
    )
    ax2.set_xlim((-0.5, len(conditions) - 0.5))
    ax2.set_title('Coding of intransitive sentences\nwith non-volitional verbs at test')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Mean proportion with overtly-coded argument')

    if input('Save results? (y/n) ').strip() == 'y':
        fn = input('Where to? (current working directory:' + os.getcwd() + ') ').strip()
        with open(fn, 'wb') as f:
            pickle.dump(data, f)
        print('The results have been saved.')
    else:
        print('The results have not been saved.')

    return data
    
