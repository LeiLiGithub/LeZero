is_android = False # 编译Android App时应当开启

from lezero.core import Variable
from lezero.core import Function
from lezero.core import Parameter
from lezero.layers import Layer

from lezero.core import using_config
from lezero.core import no_grad
from lezero.core import as_array
from lezero.core import as_variable
from lezero.core import setup_variable

from lezero.models import Model
from lezero.models import MLP
from lezero.dataloaders import DataLoader


from lezero.utils import get_file

from lezero.transforms import Compose, Flatten, ToFloat, Normalize

setup_variable()