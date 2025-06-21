from text_utils import *
from utils import CustomConstraint, mask_ents

from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapWordNet


transformation = WordSwapWordNet()

constraints = [CustomConstraint()]

augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.3,
    transformations_per_example=1,

)

# sentence = "The derivative of the <ENT> is <EXPRESSION> using the chain <ENT>."
# augmented = augmenter.augment(sentence)

# print(augmented)
