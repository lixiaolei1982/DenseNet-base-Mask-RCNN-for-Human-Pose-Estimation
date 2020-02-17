
import keras.backend as backend
SMOOTH = 1e-5


# ----------------------------------------------------------------

#   Helpers

# ----------------------------------------------------------------


def _gather_channels(x, indexes):
    """Slice tensor along channels axis by given indexes"""

    if backend.image_data_format() == 'channels_last':

        x = backend.permute_dimensions(x, (3, 0, 1, 2))

        x = backend.gather(x, indexes)

        x = backend.permute_dimensions(x, (1, 2, 3, 0))

    else:

        x = backend.permute_dimensions(x, (1, 0, 2, 3))

        x = backend.gather(x, indexes)

        x = backend.permute_dimensions(x, (1, 0, 2, 3))

    return x


def get_reduce_axes(per_image):

    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]

    if not per_image:
        axes.insert(0, 0)

    return axes


def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""

    if indexes is None:

        return xs

    elif isinstance(indexes, (int)):

        indexes = [indexes]

    xs = [_gather_channels(x, indexes=indexes) for x in xs]

    return xs


def round_if_needed(x, threshold):


    if threshold is not None:
        x = backend.greater(x, threshold)

        x = backend.cast(x, backend.floatx())

    return x


def average(x, per_image=False, class_weights=None):


    if per_image:
        x = backend.mean(x, axis=0)

    if class_weights is not None:
        x = x * class_weights

    return backend.mean(x)


# ----------------------------------------------------------------

#   Metric Functions

# ----------------------------------------------------------------


def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient

    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the

    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,

    and is defined as the size of the intersection divided by the size of the union of the sample sets:



    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}



    Args:

        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)

        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

        class_weights: 1. or list of class weights, len(weights) = C

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

        smooth: value to avoid division by zero

        per_image: if ``True``, metric is calculated as mean over images in batch (B),

            else over whole batch

        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round



    Returns:

        IoU/Jaccard score in range [0, 1]

     metric = iou_score()

        model.compile('SGD', loss=loss, metrics=[metric])

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index



    """



    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    pr = round_if_needed(pr, threshold)

    axes = get_reduce_axes(per_image)

    # score calculation

    intersection = backend.sum(gt * pr, axis=axes)

    union = backend.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)

    score = average(score, per_image, class_weights)

    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    #f1_score (beta=1)

    #f2_score (beta=2)
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,

    where an F-score reaches its best value at 1 and worst score at 0.

    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.

    The formula for the F score is:



    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}

        {\beta^2 \cdot precision + recall}



    The formula in terms of *Type I* and *Type II* errors:



    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}





    where:

        TP - true positive;

        FP - false positive;

        FN - false negative;



    Args:

        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)

        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

        class_weights: 1. or list of class weights, len(weights) = C

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

        beta: f-score coefficient

        smooth: value to avoid division by zero

        per_image: if ``True``, metric is calculated as mean over images in batch (B),

            else over whole batch

        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round



    Returns:

        F-score in range [0, 1]



    """



    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    pr = round_if_needed(pr, threshold)

    axes = get_reduce_axes(per_image)

    # calculate score

    tp = backend.sum(gt * pr, axis=axes)

    fp = backend.sum(pr, axis=axes) - tp

    fn = backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)


    score = average(score, per_image, class_weights)

    return score


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).



    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}



    where:

         - tp - true positives;

         - fp - false positives;



    Args:

        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)

        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

        smooth: Float value to avoid division by zero.

        per_image: If ``True``, metric is calculated as mean over images in batch (B),

            else over whole batch.

        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.

        name: Optional string, if ``None`` default ``precision`` name is used.



    Returns:

        float: precision score

    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    pr = round_if_needed(pr, threshold)

    axes = get_reduce_axes(per_image)

    # score calculation

    tp = backend.sum(gt * pr, axis=axes)

    fp = backend.sum(pr, axis=axes) - tp

    score = (tp + smooth) / (tp + fp + smooth)

    score = average(score, per_image, class_weights)

    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r"""Calculate recall between the ground truth (gt) and the prediction (pr).



    .. math:: F_\beta(tp, fn) = \frac{tp} {(tp + fn)}



    where:

         - tp - true positives;

         - fp - false positives;



    Args:

        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)

        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

        smooth: Float value to avoid division by zero.

        per_image: If ``True``, metric is calculated as mean over images in batch (B),

            else over whole batch.

        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.

        name: Optional string, if ``None`` default ``precision`` name is used.



    Returns:

        float: recall score

    """

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    pr = round_if_needed(pr, threshold)

    axes = get_reduce_axes(per_image)

    tp = backend.sum(gt * pr, axis=axes)

    fn = backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)

    score = average(score, per_image, class_weights)

    return score


# ----------------------------------------------------------------

#   Loss Functions

# ----------------------------------------------------------------


def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None, **kwargs):
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    # scale predictions so that the class probas of each sample sum to 1

    axis = 3 if backend.image_data_format() == 'channels_last' else 1

    pr /= backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's

    pr = backend.clip(pr, backend.epsilon(), 1 - backend.epsilon())

    # calculate loss

    output = gt * backend.log(pr) * class_weights

    return - backend.mean(output)


def binary_crossentropy(gt, pr, **kwargs):
    backend = kwargs['backend']

    return backend.mean(backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None):
    r"""Implementation of Focal Loss from the paper in multiclass classification



    Formula:

        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)



    Args:

        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)

        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

        alpha: the same as weighting factor in balanced cross entropy, default 0.25

        gamma: focusing parameter for modulating factor (1-p), default 2.0

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.



    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    # clip to prevent NaN's and Inf's

    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss

    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))
    

    return backend.mean(loss)




def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25):
    r"""Implementation of Focal Loss from the paper in binary classification



    Formula:

        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \

               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)



    Args:

        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)

        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

        alpha: the same as weighting factor in balanced cross entropy, default 0.25

        gamma: focusing parameter for modulating factor (1-p), default 2.0



    """
    # clip to prevent NaN's and Inf's

    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    loss_1 = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    loss_0 = - (1 - gt) * (alpha * backend.pow((pr), gamma) * backend.log(1 - pr))

    loss = backend.mean(loss_0 + loss_1)

    return loss

class KerasObject:
    def __init__(self, name=None):
        self._name = name

    @property

    def __name__(self):

        if self._name is None:

            return self.__class__.__name__

        return self._name

    @property

    def name(self):

        return self.__name__

    @name.setter

    def name(self, name):

        self._name = name


class Loss(KerasObject):



    def __add__(self, other):

        if isinstance(other, Loss):

            return SumOfLosses(self, other)

        else:

            raise ValueError('Loss should be inherited from `Loss` class')



    def __radd__(self, other):

        return self.__add__(other)



    def __mul__(self, value):

        if isinstance(value, (int, float)):

            return MultipliedLoss(self, value)

        else:

            raise ValueError('Loss should be inherited from `BaseLoss` class')



    def __rmul__(self, other):

        return self.__mul__(other)





class MultipliedLoss(Loss):



    def __init__(self, loss, multiplier):



        # resolve name

        if len(loss.__name__.split('+')) > 1:

            name = '{}({})'.format(multiplier, loss.__name__)

        else:

            name = '{}{}'.format(multiplier, loss.__name__)

        super().__init__(name=name)

        self.loss = loss

        self.multiplier = multiplier



    def __call__(self, gt, pr):

        return self.multiplier * self.loss(gt, pr)





class SumOfLosses(Loss):



    def __init__(self, l1, l2):

        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)

        super().__init__(name=name)

        self.l1 = l1

        self.l2 = l2



    def __call__(self, gt, pr):

        return self.l1(gt, pr) + self.l2(gt, pr)






class JaccardLoss(Loss):

    r"""Creates a criterion to measure Jaccard loss:



    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}



    Args:

        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

        per_image: If ``True`` loss is calculated for each image in batch and then averaged,

            else loss is calculated for the whole batch.

        smooth: Value to avoid division by zero.



    Returns:

         A callable ``jaccard_loss`` instance. Can be used in ``model.compile(...)`` function

         or combined with other losses.



    Example:



    .. code:: python



        loss = JaccardLoss()

        model.compile('SGD', loss=loss)

    """



    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):

        super().__init__(name='jaccard_loss')

        self.class_weights = class_weights if class_weights is not None else 1

        self.class_indexes = class_indexes

        self.per_image = per_image

        self.smooth = smooth



    def __call__(self, gt, pr):

        return 1 - iou_score(

            gt,

            pr,

            class_weights=self.class_weights,

            class_indexes=self.class_indexes,

            smooth=self.smooth,

            per_image=self.per_image,

            threshold=None

        )





class DiceLoss(Loss):

    r"""Creates a criterion to measure Dice loss:



    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}

        {\beta^2 \cdot precision + recall}



    The formula in terms of *Type I* and *Type II* errors:



    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}



    where:

         - tp - true positives;

         - fp - false positives;

         - fn - false negatives;



    Args:

        beta: Float or integer coefficient for precision and recall balance.

        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

        per_image: If ``True`` loss is calculated for each image in batch and then averaged,

        else loss is calculated for the whole batch.

        smooth: Value to avoid division by zero.



    Returns:

        A callable ``dice_loss`` instance. Can be used in ``model.compile(...)`` function`

        or combined with other losses.



    Example:



    .. code:: python



        loss = DiceLoss()

        model.compile('SGD', loss=loss)

    """



    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')

        self.beta = beta

        self.class_weights = class_weights if class_weights is not None else 1

        self.class_indexes = class_indexes

        self.per_image = per_image

        self.smooth = smooth



    def __call__(self, gt, pr):

        return 1 - f_score(

            gt,

            pr,

            beta=self.beta,

            class_weights=self.class_weights,

            class_indexes=self.class_indexes,

            smooth=self.smooth,

            per_image=self.per_image,

            threshold=None


        )





class BinaryCELoss(Loss):

    """Creates a criterion that measures the Binary Cross Entropy between the

    ground truth (gt) and the prediction (pr).



    .. math:: L(gt, pr) = - gt \cdot \log(pr) - (1 - gt) \cdot \log(1 - pr)



    Returns:

        A callable ``binary_crossentropy`` instance. Can be used in ``model.compile(...)`` function

        or combined with other losses.



    Example:



    .. code:: python



        loss = BinaryCELoss()

        model.compile('SGD', loss=loss)

    """

    def __init__(self):
        super().__init__(name='binary_crossentropy')



    def __call__(self, gt, pr):

        return binary_crossentropy(gt, pr)





class CategoricalCELoss(Loss):

    """Creates a criterion that measures the Categorical Cross Entropy between the

    ground truth (gt) and the prediction (pr).



    .. math:: L(gt, pr) = - gt \cdot \log(pr)



    Args:

        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.



    Returns:

        A callable ``categorical_crossentropy`` instance. Can be used in ``model.compile(...)`` function

        or combined with other losses.



    Example:



    .. code:: python



        loss = CategoricalCELoss()

        model.compile('SGD', loss=loss)

    """



    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights if class_weights is not None else 1

        self.class_indexes = class_indexes



    def __call__(self, gt, pr):

        return categorical_crossentropy(

            gt,

            pr,

            class_weights=self.class_weights,

            class_indexes=self.class_indexes


        )





class CategoricalFocalLoss(Loss):

    r"""Creates a criterion that measures the Categorical Focal Loss between the

    ground truth (gt) and the prediction (pr).



    .. math:: L(gt, pr) = - gt \cdot \alpha \cdot (1 - pr)^\gamma \cdot \log(pr)



    Args:

        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.

        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.

        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.



    Returns:

        A callable ``categorical_focal_loss`` instance. Can be used in ``model.compile(...)`` function

        or combined with other losses.



    Example:



        .. code:: python



            loss = CategoricalFocalLoss()

            model.compile('SGD', loss=loss)

    """



    def __init__(self, alpha=0.25, gamma=2., class_indexes=None):
        super().__init__(name='focal_loss')
        self.alpha = alpha

        self.gamma = gamma

        self.class_indexes = class_indexes



    def __call__(self, gt, pr):

        return categorical_focal_loss(

            gt,

            pr,

            self.alpha,

            self.gamma,

            class_indexes=self.class_indexes

        )





class BinaryFocalLoss(Loss):

    r"""Creates a criterion that measures the Binary Focal Loss between the

    ground truth (gt) and the prediction (pr).



    .. math:: L(gt, pr) = - gt \alpha (1 - pr)^\gamma \log(pr) - (1 - gt) \alpha pr^\gamma \log(1 - pr)



    Args:

        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.

        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.



    Returns:

        A callable ``binary_focal_loss`` instance. Can be used in ``model.compile(...)`` function

        or combined with other losses.



    Example:



    .. code:: python



        loss = BinaryFocalLoss()

        model.compile('SGD', loss=loss)

    """



    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha

        self.gamma = gamma



    def __call__(self, gt, pr):

        return binary_focal_loss(gt, pr, self.alpha, self.gamma)





# aliases

jaccard_loss = JaccardLoss()

dice_loss = DiceLoss()



Binary_focal_loss = BinaryFocalLoss()

Categorical_focal_loss = CategoricalFocalLoss()



Binary_crossentropy = BinaryCELoss()

Categorical_crossentropy = CategoricalCELoss()



# loss combinations

bce_dice_loss = Binary_crossentropy + dice_loss

bce_jaccard_loss = Binary_crossentropy + jaccard_loss



cce_dice_loss = Categorical_crossentropy + dice_loss

cce_jaccard_loss = Categorical_crossentropy + jaccard_loss



binary_focal_dice_loss = Binary_focal_loss + dice_loss

binary_focal_jaccard_loss = Binary_focal_loss + jaccard_loss



categorical_focal_dice_loss = Categorical_focal_loss + dice_loss

categorical_focal_jaccard_loss = Categorical_focal_loss + jaccard_loss