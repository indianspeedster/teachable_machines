::: {.cell .markdown}
# Using edge devices for CPU-based inference

:::


::: {.cell .markdown}

Machine learning models are most often trained in the "cloud", on powerful centralized servers with specialized resources (like GPU acceleration) for training machine learning models. 


However, for a variety of reasons including privacy, latency, and network connectivity or bandwidth constraints, it is often preferable to *use* these models (i.e. do inference) at "edge" devices located wherever the input data is/where the model's prediction is going to be used. 


These edge devices are less powerful and typically lack any special acceleration, so the inference time (the time from when the input is fed to the model, until the model outputs its prediction) may not be as fast as it would be on a cloud server - but we avoid having to send the input data to the cloud and then sending the prediction back.

:::

::: {.cell .markdown}

This notebook assumes you already have a "lease" available for a device on the CHI@Edge testbed. Then, it will show you how to:

* launch a "container" on that device
* attach an IP address to the container, so that you can access it over SSH
* transfer files to and from the container
* use a pre-trained image classification model to do inference on the edge device
* delete the container

:::

::: {.cell .markdown}
## Launch a container on an edge device

We will start by preparing our environment in this notebook, then launching a container on an edge device using our pre-existing lease.

:::

::: {.cell .markdown}

First, we load some required libraries:

:::

::: {.cell .code}
``` python
import chi
from chi import container
from chi import lease
import datetime
import os
```
:::

::: {.cell .markdown}

We indicate that we're going to use the CHI@Edge site. We also need to specify the name of the Chameleon "project" that this experiment is part of. The project name will have the format "CHI-XXXXXX", where the last part is a 6-digit number, and you can find it on your [user dashboard](https://chameleoncloud.org/user/dashboard/).

In the cell below, replace the project ID with your *own* project ID, then run the cell.

:::

::: {.cell .code}
``` python
chi.use_site("CHI@Edge")
chi.set("project_name", "CHI-XXXXXX")
```
:::

::: {.cell .markdown}

Next, we'll specify the lease ID. This notebook assumes you already have a "lease" for a device on CHI@Edge. To get the ID of this lease,

* Vist the CHI@Edge ["reservations" page](https://chi.edge.chameleoncloud.org/project/leases/).
* Click on the lease name.
* On the following page, look for the value next to the word "Id" in the "Lease" section.

Fill in the lease ID inside the quotation marks in the following cell, then run the cell.


:::

::: {.cell .code}
``` python
lease_id ="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```
:::

::: {.cell .markdown}

Now, we are ready to launch a container!

-   **Container** : A container is like a logical "box" that holds everything needed to run an application. It includes the application itself, along with all the necessary prerequisite software, files, and settings it needs to work properly. 
-   **Image** : An image is like a pre-packaged "starting point" for a container. On CHI@Edge, we can use any image that is built for the ARM64 architecture - e.g. anything on [this list](https://hub.docker.com/search?type=image&architecture=arm64&q=). In this example, we're going to run a machine learning application written in Python, so we will use the `python:3.9-slim` image as a starting point for our container. This is a lightweight installation of the Debian Linux operating system with Python pre-installed.

When we create the container, we could also specify some additional arguments: 

-   `workdir`: the "working directory" - location in the container's filesystem from which any commands we specify will run.
-   `exposed_ports`: if we run any applications inside the container that need to accept incoming requests from a network, we will need to export a "port" number for those incoming requests. Any requests to that port number will be forwarded to this container.
-   `command`: if we want to execute a specific command immediately on starting the container, we can specify that as well.

For this particular experiment, we'll specify that port 22 - which is used for SSH access - should be exposed. 

Also, since we do not specify a `command` to run, we will further specify `interactive = True` - that it should open an interactive Python session - otherwise the container will immediately stop after it is started, because it has no "work" to do.

:::

::: {.cell .markdown}

First, we'll specify the name for our container - we'll include our username and the experiment name in the container name, so that it will be easy to identify our container in the CHI@Edge web interface.

:::

::: {.cell .code}
``` python
username = os.environ.get("USER")
expname = "edge-cpu"
# set a name for the container
# Note that underscore characters _ are not allowed - we replace each _ with a -
container_name = f"{username}-{expname}".replace("_","-")
```
:::

::: {.cell .markdown}

Then, we can create the container!

:::

::: {.cell .code}
``` python
try:
    my_container = container.create_container(
        container_name,
        image="python:3.9-slim",
        reservation_id=lease.get_device_reservation(lease_id),
        interactive=True,
        exposed_ports=[22],
        platform_version=2,
    )
except RuntimeError as ex:
    print(ex)
    print(f"Please stop and/or delete {container_name} and try again")
else:
    print(f"Successfully created container: {container_name}!")
```
:::

::: {.cell .markdown}

The next cell waits for the container to be active - when it is, it will print some output related to the container state.
:::

::: {.cell .code}
```python
# wait until container is ready to use
container.wait_for_active(my_container.uuid)
```
:::



::: {.cell .markdown}

Once the container is created, you should be able to see it and monitor its status on the [CHI@Edge web interface](https://chi.edge.chameleoncloud.org/project/container/containers). (If there was any problem while creating the container, you can also delete the container from that interface, in order to be able to try again.)

:::

::: {.cell .markdown}
## Attach an address and access your container over SSH

:::

::: {.cell .markdown}

Just as with a conventional "server" on Chameleon, we can attach an address to our container, then use SSH to access its terminal.

First, we'll attach an address:

:::

::: {.cell .code}
``` python
public_ip = container.associate_floating_ip(my_container.uuid)
```
:::

::: {.cell .markdown}

Then, we need to install an SSH server on the container - it is not pre-installed on the image we selected.  We can use the `container.execute()` function to run commands inside the container, in order to install the SSH server.

:::


::: {.cell .code}
```python
container.execute(my_container.uuid, 'apt update')
container.execute(my_container.uuid, 'apt -y install openssh-server')
```
:::

::: {.cell .markdown}

There is one more necessary step before we can access the container over SSH - we need to make sure our key is installed on the container. Here, we will upload the key from the Jupyter environment, and make sure it is configured with the appropriate file permissions:
:::


::: {.cell .code}
```python
!mkdir -p tmp_keys
!cp /work/.ssh/id_rsa.pub tmp_keys/authorized_keys
```
:::


::: {.cell .code}
```python
container.execute(my_container.uuid, 'mkdir -p /root/.ssh')
container.upload(my_container.uuid, "./tmp_keys", "/root/.ssh")
container.execute(my_container.uuid, 'chown root /root/.ssh')
container.execute(my_container.uuid, 'chown root /root/.ssh/authorized_keys')
container.execute(my_container.uuid, 'chmod go-w /root')
container.execute(my_container.uuid, 'chmod 700 /root/.ssh')
container.execute(my_container.uuid, 'chmod 600 /root/.ssh/authorized_keys')
```
:::


::: {.cell .markdown}

Start the SSH server in the container. The following cell should print "sshd is running". It it's not running, it can be an indication that the SSH server was not fully installed; wait a minute or two and then try this cell again:

:::


::: {.cell .code}
```python
container.execute(my_container.uuid, 'service ssh start')
container.execute(my_container.uuid, 'service ssh status')
```
:::



::: {.cell .markdown}

Now we can open a terminal in the Jupyter interface to access the container over SSH, using the SSH command that is printed by the following cell:

:::

::: {.cell .code}
``` python
print("ssh root@%s" % public_ip)
```
:::


::: {.cell .markdown}
## Transfering files to the container

Later in this notebook, we'll run an image classification model - a model that accepts an image as input and "predicts" the name of the object in the image - inside the container. To do this, we'll need to upload some files to the container:

* an already-trained model
* a list of labels - this maps the integer values "predicted" by the model to human readable object names
* a sample image
* and Python code to load the model and make a prediction on the image

These are all contained in the `image_model` directory. We can upload them to the container using the `container.upload` function, and specify the source directory (in the Jupyter environment) and destination directory (on the container).


:::

::: {.cell .code}
``` python
container.execute(my_container.uuid, 'mkdir -p /root/image_model')
container.upload(my_container.uuid, "./image_model", "/root/image_model")
```
:::


::: {.cell .markdown}
## Use a pre-trained image classification model to do inference

Now, we can use the model we uploaded to the container, and do inference - make a prediction - *on* the container. 


In this example, we will use a machine learning model that is specifically designed for inference on resource-constrained edge devices. In general, there are several strategies to reduce inference time on edge devices:

* **Model design**: models meant for inference on edge devices are often designed specifically to reduce memory and/or inference time. The model in this example is a MobileNet, which like many image classification models uses a *convolution* operation to process its input, but MobileNets use a kind of convolution that is much faster and requires fewer operations than a "standard" convolution.
* **Model Compression**: another approach to faster inference on edge devices is model compression, a group of techniques that try to reduce the size of the model without affecting its accuracy. The model in this example is a quantized model, which means that the numeric parameters in the model are represented using fewer bits than in a "standard" model. These quantized parameters can also be processed using faster mathematical operations, potentially improving the inference time.
* **Hardware Acceleration**: a third popular technique to improving inference time at the edge is with hardware acceleration - using specialized computer chips, GPUs, or TPUs that can perform the operations involved in inference very fast. In this example, though, we are going to use CPU-based inference, which means that there is no hardware acceleration.


--- 

> For more information about these strategies, see:
> J. Chen and X. Ran, "Deep Learning With Edge Computing: A Review," in Proceedings of the IEEE, vol. 107, no. 8, pp. 1655-1674, Aug. 2019, doi: 10.1109/JPROC.2019.2921977. https://ieeexplore.ieee.org/document/8763885 


:::


::: {.cell .markdown}
First, we need to install a couple of Python libraries in the container:

* `tflite` is a library specifically designed for machine learning inference on edge devices.
* `Pillow` is used for image processing.

:::

::: {.cell .code}
```python
container.execute(my_container.uuid, 'pip install tflite-runtime Pillow')
```
:::

::: {.cell .markdown}

Then, we can execute the machine learning model! We will ask it to make a prediction for the following image:

:::


::: {.cell .code}
```python
from IPython.display import Image
Image('image_model/parrot.jpg') 
```
:::

::: {.cell .code}
```python
result = container.execute(my_container.uuid, 'python /root/image_model/model.py')
print(result['output'])
```
:::

::: {.cell .markdown}

Make a note of the time it took to generate the prediction - would this inference time be acceptable for all applications? Also make a note of the model's three best "guesses" regarding the label of the image - is the prediction accurate?

:::

::: {.cell .markdown}
## Delete the container

Finally, we should stop and delete our container so that others can create new containers using the same lease. To delete our container, we can run the following cell:

:::

::: {.cell .code}
```python
container.destroy_container(my_container.uuid)
```
:::


::: {.cell .markdown}
Also free up the IP that you we attached to the container, now that it is no longer in use:
:::



::: {.cell .code}
```python
ip_details = chi.network.get_floating_ip(public_ip)
chi.neutron().delete_floatingip(ip_details["id"])
```
:::
