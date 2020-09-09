from functools import partial
import sys
from typing import Tuple

from bokeh.models import Arrow, Button, ColorBar, ColumnDataSource, CustomJS, DataTable
from bokeh.models import Div, Dropdown, HoverTool, LinearColorMapper, NumberFormatter
from bokeh.models import Slider, TableColumn, Toggle, VeeHead
from bokeh.palettes import Magma256
from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import optimizers

def init() -> None:

    tf.random.set_seed(0)

    tensors['X'] = tf.random.normal((sliders['num_samples'].value, 2), mean=0, stddev=1)

    for target in ['true', 'model']:
        weight_0 = sliders[f'weight_{target}_0'].value
        weight_1 = sliders[f'weight_{target}_1'].value
        tensors[f'weight_{target}'] = tf.Variable(
            [[weight_0], [weight_1]],
            dtype=tf.dtypes.float32
        )
        tensors[f'y_{target}'] = pass_neuron(
            tensors['X'],
            tensors[f'weight_{target}']
        )

    tensors['loss'] = loss_function(tensors['y_true'], tensors['y_model'])

    tensors['y_domain'] = pass_neuron(tensors['X'], tensors['weight_domain'])
    tensors['loss_surface'] = loss_function(tensors['y_true'], tensors['y_domain'])

    data_sources['true'].data = {
        'weight_0': [sliders['weight_true_0'].value],
        'weight_1': [sliders['weight_true_1'].value]
    }
    data_sources['history'].data = {
        'weight_0': [sliders['weight_model_0'].value],
        'weight_1': [sliders['weight_model_1'].value]
    }
    data_sources['now'].data = {
        'weight_0': [sliders['weight_model_0'].value],
        'weight_1': [sliders['weight_model_1'].value],
        'loss': [tensors['loss'].numpy()[0]],
        'epoch': [0]
    }

    color_mapper = fig.select_one({'name': 'color_mapper'})
    color_mapper.low = np.min(tensors['loss_surface'].numpy())
    color_mapper.high = np.max(tensors['loss_surface'].numpy())

    loss_surface = [tensors['loss_surface'].numpy().reshape(tile_shape)]
    fig.select_one({'name': 'loss'}).data_source.data['image'] = loss_surface

    fig.center = fig.center[0:2]

    return

def create_batches() -> Tuple[tf.Tensor, tf.Tensor]:
    num_samples_in_batches = [sliders['batch_size'].value] * (
        sliders['num_samples'].value //
        sliders['batch_size'].value
    )

    if sliders['num_samples'].value % sliders['batch_size'].value != 0:
        num_samples_in_batches.append(
            sliders['num_samples'].value % sliders['batch_size'].value
        )

    indices = tf.range(start=0, limit=sliders['num_samples'].value, dtype=tf.int32)
    tf.random.set_seed(data_sources['now'].data['epoch'][0] + 1)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(tensors['X'], shuffled_indices, axis=0)
    shuffled_y = tf.gather(tensors['y_true'], shuffled_indices, axis=0)

    batches_x = tf.split(shuffled_x, num_samples_in_batches, axis=0)
    batches_y = tf.split(shuffled_y, num_samples_in_batches, axis=0)

    return batches_x, batches_y

def update() -> None:
    if toggles['pause'].active:
        return

    ds_now = data_sources['now'].data
    ds_history = data_sources['history'].data

    if ds_now['epoch'][0] + 1 > sliders['max_epoch'].value:
        return

    batches_x, batches_y = create_batches()

    for batch_x, batch_y in zip(batches_x, batches_y):
        weight_previous = tensors['weight_model']
        weight_updated = gradient_descent(batch_y, batch_x, weight_previous)

        arrow_head = VeeHead(
            size=5,
            fill_color='darkgrey',
            line_color='darkgrey'
        )
        arrow = Arrow(
            end=arrow_head,
            line_color='darkgrey',
            x_start=ds_history['weight_0'][-1],
            y_start=ds_history['weight_1'][-1],
            x_end=weight_updated.numpy()[0, 0],
            y_end=weight_updated.numpy()[1, 0]
        )
        fig.add_layout(arrow)

        loss = loss_function(
            tensors['y_true'],
            pass_neuron(tensors['X'], weight_updated)
        ).numpy()[0]

        ds_history['weight_0'] += [weight_updated.numpy()[0, 0]]
        ds_history['weight_1'] += [weight_updated.numpy()[1, 0]]

        ds_now['loss'] = [loss]
        ds_now['weight_0'] = [weight_updated.numpy()[0, 0]]
        ds_now['weight_1'] = [weight_updated.numpy()[1, 0]]

        tensors['weight_model'] = weight_updated

    ds_now['epoch'] = [ds_now['epoch'][0] + 1]

    for arrow in fig.center[2:-sliders['num_samples'].value]:
        arrow.line_color = 'lightgrey'
        arrow.end.fill_color = 'lightgrey'
        arrow.end.line_color = 'lightgrey'

    return

def pass_neuron(X: tf.Tensor, weight: tf.Variable) -> tf.Tensor:
    return selectors['activation']['Activate'](tf.matmul(X, weight))

def loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=0)

def gradient_descent(y: tf.Tensor, X: tf.Tensor, weight: tf.Variable) ->  tf.Variable:
    weight = tf.Variable(weight)
    optimizer = selectors['optimizer']['Activate']
    optimizer.learning_rate = (
        selectors['eta_modifier']['Activate'] *
        sliders['eta'].value
    )
    optimizer.minimize(
        lambda: tf.reduce_mean(tf.pow(y - pass_neuron(X, weight), 2)),
        var_list=[weight]
    )
    return tf.Variable(weight)

def restart() -> None:
    init()
    sliders['batch_size'].end = sliders['num_samples'].value
    if sliders['batch_size'].value > sliders['num_samples'].value:
        sliders['batch_size'].value = sliders['num_samples'].value

def change_params(attr, new, old) -> None:
    restart()

def callback_pause(event, ref):
    if event:
        ref.button_type = "warning"
        ref.label = 'Run'
    else:
        ref.button_type = "success"
        ref.label = 'Pause'

def callback_dropdown(event, ref, selector):
    ref.label = event.item
    selector['Activate'] = selector[event.item]
    try:
        selectors['eta_modifier']['Activate'] = selectors['eta_modifier'][event.item]
    except:
        pass

    restart()

header = Div(text='<h1>Visualization for Optimizers', width=500, height=50)

sliders = {
    'weight_true_0': Slider(
        start=-20,
        end=20,
        value=5,
        step=0.1,
        title='Ground Truth Weight 1',
        align='center',
        width=245
    ),
    'weight_true_1': Slider(
        start=-20,
        end=20,
        value=5,
        step=0.1,
        title='Ground Truth Weight 2',
        align='center',
        width=245
    ),
    'weight_model_0': Slider(
        start=-20,
        end=20,
        value=-5,
        step=0.1,
        title='Initial Weight 1',
        align='center',
        width=245
    ),
    'weight_model_1': Slider(
        start=-20,
        end=20,
        value=-5,
        step=0.1,
        title='Initial Weight 2',
        align='center',
        width=245
    ),
    'num_samples': Slider(
        start=10,
        end=20,
        value=10,
        step=2,
        title='Number of Samples',
        align='center',
        width=500
    ),
    'batch_size': Slider(
        start=5,
        end=10,
        value=5,
        step=1,
        title='Batch Size',
        align='center',
        width=500
    ),
    'max_epoch': Slider(
        start=1,
        end=100,
        value=25,
        step=1,
        title='Number of Epochs',
        align='center',
        width=500
    ),
    'eta': Slider(
        start=0.01,
        end=1.0,
        value=0.25,
        step=0.01,
        title='Learning Rate',
        align='center',
        width=500
    )
}

for slider in sliders.values():
    slider.on_change('value', change_params)

buttons = {
    'restart': Button(label='Restart', align='center', width=165),
    'stop': Button(label='Stop', align='center', button_type='danger', width=165)
}

buttons['restart'].on_click(restart)
buttons['stop'].js_on_click(CustomJS(code='window.close()'))
buttons['stop'].on_click(lambda: sys.exit())

toggles = {
    'pause': Toggle(label='Pause', align='center', button_type='success', width=165)
}

toggles['pause'].on_click(partial(callback_pause, ref=toggles['pause']))

menus = {
    'activation': [
        ('Sigmoid', 'Sigmoid'),
        ('Tanh', 'Tanh'),
        ('ReLU', 'ReLU'),
        ('Swish', 'Swish'),
        ('Custom', 'Custom')
    ],
    'optimizer': [
        ('SGD', 'SGD'),
        ('AdaGrad', 'AdaGrad'),
        ('RMSProp', 'RMSProp'),
        ('Momemtum', 'Momemtum'),
        ('Nesterov', 'Nesterov'),
        ('Adam', 'Adam'),
        ('Nadam', 'Nadam')
    ]
}

selectors = {
    'activation': {
        'Sigmoid': activations.sigmoid,
        'Tanh': activations.tanh,
        'ReLU': activations.relu,
        'Swish': lambda x: x * activations.sigmoid(x),
        'Custom': lambda x: tf.math.log(tf.pow(tf.abs(x), 2 + tf.math.sign(x)) + 1),
        'Activate': activations.sigmoid
    },
    'optimizer': {
        'SGD': optimizers.SGD(),
        'AdaGrad': optimizers.Adagrad(),
        'RMSProp': optimizers.RMSprop(),
        'Momemtum': optimizers.SGD(momentum=0.9),
        'Nesterov': optimizers.SGD(momentum=0.9, nesterov=True),
        'Adam': optimizers.Adam(),
        'Nadam': optimizers.Nadam(),
        'Activate': optimizers.SGD()
    },
    'eta_modifier': {
        'SGD': 50,
        'AdaGrad': 50,
        'RMSProp': 1,
        'Momemtum': 50,
        'Nesterov': 50,
        'Adam': 5,
        'Nadam': 5,
        'Activate': 50
    }
}

dropdowns = {
    'activation': Dropdown(
        label='Sigmoid',
        menu=menus['activation'],
        width=245
    ),
    'optimizer': Dropdown(
        label='SGD',
        menu=menus['optimizer'],
        width=245
    )
}

for name, dropdown in dropdowns.items():
    dropdown.on_click(partial(
        callback_dropdown,
        ref=dropdown,
        selector=selectors[name]
    ))

configs = {
    'num_tiles': 500
}

weight_0, weight_1 = np.meshgrid(
    np.linspace(-20., 20., num=configs['num_tiles']),
    np.linspace(-20., 20., num=configs['num_tiles'])
)

tf.random.set_seed(0)

tensors = {
    'weight_domain': tf.Variable(
        np.array([weight_0.ravel(), weight_1.ravel()], dtype=np.float32)
    )
}

data_sources = {
    'true': ColumnDataSource(data={
        'weight_0': [sliders['weight_true_0'].value],
        'weight_1': [sliders['weight_true_1'].value]
    }),
    'history': ColumnDataSource(data={
        'weight_0': [sliders['weight_model_0'].value],
        'weight_1': [sliders['weight_model_1'].value]
    }),
    'now': ColumnDataSource(data={
        'weight_0': [sliders['weight_model_0'].value],
        'weight_1': [sliders['weight_model_1'].value],
        'loss': [np.nan],
        'epoch': [0]
    })
}

color_mapper = LinearColorMapper(
    Magma256[::-1],
    name='color_mapper',
    low=0,
    high=1
)

color_bar = ColorBar(
    color_mapper=color_mapper,
    border_line_color=None,
    location=(0, 0),
    title_standoff=8,
    label_standoff=8,
    title='MSE',
    scale_alpha=0.5
)

fig = figure(
    title='Optimizers',
    tools=[],
    plot_width=1024,
    plot_height=768,
    x_axis_label='Weight 1',
    y_axis_label='Weight 2',
    x_range=(-20, 20),
    y_range=(-20, 20)
)
fig.add_layout(color_bar, 'right')
tile_shape = (configs['num_tiles'], configs['num_tiles'])

image = fig.image(
    image=[np.zeros(tile_shape)],
    x=-20, y=-20, dw=40, dh=40,
    alpha=0.5,
    name='loss',
    color_mapper=color_mapper
)
tooltips = [("weight 0", "$x"), ("weight 1", "$y"), ("loss", "@image")]
fig.add_tools(HoverTool(renderers=[image], tooltips=tooltips))
fig.x(
    x='weight_0',
    y='weight_1',
    fill_alpha=0.0,
    line_color='red',
    size=20,
    source=data_sources['true']
)
fig.circle(
    x='weight_0',
    y='weight_1',
    fill_alpha=0.5,
    color='lightgrey',
    size=12,
    source=data_sources['history']
)
fig.circle(
    x='weight_0',
    y='weight_1',
    fill_alpha=1.0,
    color='black',
    size=12,
    source=data_sources['now']
)

int_format = NumberFormatter(text_align='right')
float_format = NumberFormatter(text_align='right', format='0,0.0000')
columns = [
    TableColumn(field="epoch", title="Epoch", formatter=int_format),
    TableColumn(field="weight_0", title="Weight 1", formatter=float_format),
    TableColumn(field="weight_1", title="Weight 2", formatter=float_format),
    TableColumn(field="loss", title="Loss", formatter=float_format)
]
datatable = DataTable(
    source=data_sources['now'],
    columns=columns,
    width=500, height=50,
    index_position=None,
    align='center'
)

init()

tools = column(
    header,
    row(dropdowns['activation'], dropdowns['optimizer']),
    row(sliders['weight_true_0'], sliders['weight_true_1']),
    row(sliders['weight_model_0'], sliders['weight_model_1']),
    sliders['num_samples'],
    sliders['batch_size'],
    sliders['max_epoch'],
    sliders['eta'],
    datatable,
    row(buttons['restart'], toggles['pause'], buttons['stop'])
)

layout = row(column(fig), tools)
curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 1000)
