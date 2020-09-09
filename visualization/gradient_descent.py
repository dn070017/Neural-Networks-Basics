#%% 
from functools import partial
import sys
from typing import Tuple

from bokeh.models import Arrow, Button, ColorBar, ColumnDataSource, CustomJS, DataTable
from bokeh.models import Div, Dropdown, LinearColorMapper, NumberFormatter, Slider
from bokeh.models import TableColumn, Toggle, VeeHead
from bokeh.models.ranges import Range1d
from bokeh.palettes import Magma256
from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import optimizers

#%%
def init() -> None:

    tf.random.set_seed(0)

    tensors['X'] = tf.random.normal((sliders['num_samples'].value, 1), mean=0, stddev=1)

    for target in ['true', 'model']:
        weight = sliders[f'weight_{target}'].value
        tensors[f'weight_{target}'] = tf.Variable(
            [[weight]],
            dtype=tf.dtypes.float32
        )
        tensors[f'y_{target}'] = pass_neuron(
            tensors['X'],
            tensors[f'weight_{target}']
        )

    tensors['loss'] = loss_function(tensors['y_true'], tensors['y_model'])

    tensors['y_domain'] = pass_neuron(tensors['X'], tensors['weight_domain'])
    tensors['loss_curve'] = loss_function(tensors['y_true'], tensors['y_domain'])

    data_sources['true'].data = {
        'weight': [sliders['weight_true'].value],
        'loss': [0]
    }
    data_sources['history'].data = {
        'weight': [sliders['weight_model'].value],
        'loss': [tensors['loss'].numpy()[0]]
    }
    data_sources['now'].data = {
        'weight': [sliders['weight_model'].value],
        'loss': [tensors['loss'].numpy()[0]],
        'epoch': [0]
    }

    loss_curve = tensors['loss_curve'].numpy()
    fig.select_one({'name': 'loss'}).data_source.data['y'] = loss_curve

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

        loss = loss_function(
            tensors['y_true'],
            pass_neuron(tensors['X'], weight_updated)
        ).numpy()[0]

        arrow_head = VeeHead(
            size=5,
            fill_color='darkgrey',
            line_color='darkgrey'
        )
        arrow = Arrow(
            end=arrow_head,
            line_color='darkgrey',
            x_start=ds_history['weight'][-1],
            y_start=ds_history['loss'][-1],
            x_end=weight_updated.numpy()[0, 0],
            y_end=loss
        )
        fig.add_layout(arrow)

        ds_history['weight'] += [weight_updated.numpy()[0, 0]]
        ds_history['loss'] += [loss]
        ds_now['loss'] = [loss]
        ds_now['weight'] = [weight_updated.numpy()[0, 0]]
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

#%%
header = Div(text='<h1>Visualization for Gradient Descent', width=500, height=50)

sliders = {
    'weight_true': Slider(
        start=-20,
        end=20,
        value=10,
        step=0.1,
        title='Ground Truth Weight',
        align='center',
        width=245
    ),
    'weight_model': Slider(
        start=-20,
        end=20,
        value=-5,
        step=0.1,
        title='Initial Weight',
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

selectors = {
    'activation': {
        'Activate': lambda x: (x ** 3 + 50 * x ** 2 + 10 * x) / 1000
    },
    'optimizer': {
        'Activate': optimizers.SGD()
    },
    'eta_modifier': {
        'Activate': 1
    }
}

configs = {
    'linspace_size': 500
}

tf.random.set_seed(0)

tensors = {
    'weight_domain': tf.reshape(
        tf.Variable(tf.linspace(-20., 20., configs['linspace_size'])), 
        (1, configs['linspace_size'])
    )
}

data_sources = {
    'true': ColumnDataSource(data={
        'weight': [sliders['weight_true'].value],
        'loss': [0]
    }),
    'history': ColumnDataSource(data={
        'weight': [sliders['weight_model'].value],
        'loss': [np.nan],
    }),
    'now': ColumnDataSource(data={
        'weight': [sliders['weight_model'].value],
        'loss': [np.nan],
        'epoch': [0]
    })
}

fig = figure(
    title='Gradient Descent',
    tools=[],
    plot_width=1024,
    plot_height=768,
    x_axis_label='Weight',
    y_axis_label='Loss',
    x_range=(-20, 20),
    y_range=(0, 100)
)
fig.line(
    x=tensors['weight_domain'].numpy().reshape(-1),
    y=np.zeros(configs['linspace_size']),
    name='loss',
    line_color='darkviolet',
    line_width=2
)
fig.x(
    x='weight',
    y='loss',
    fill_alpha=0.0,
    line_color='red',
    size=20,
    source=data_sources['true']
)
fig.circle(
    x='weight',
    y='loss',
    fill_alpha=0.5,
    color='lightgrey',
    size=12,
    source=data_sources['history']
)
fig.circle(
    x='weight',
    y='loss',
    fill_alpha=1.0,
    color='black',
    size=12,
    source=data_sources['now']
)

int_format = NumberFormatter(text_align='right')
float_format = NumberFormatter(text_align='right', format='0,0.0000')
columns = [
    TableColumn(field="epoch", title="Epoch", formatter=int_format),
    TableColumn(field="weight", title="Weight", formatter=float_format),
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
    row(sliders['weight_true'], sliders['weight_model']),
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
