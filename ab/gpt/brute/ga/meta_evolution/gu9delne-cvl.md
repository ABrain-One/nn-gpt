Contact person: Dmytro Ignatov [dmytro.ignatov@uni-wuerzburg.de](mailto:dmytro.ignatov@uni-wuerzburg.de)

Once you have accessed the CV cluster to start using it, log in to the master node **132.187.14.67** with SSH command:

**ssh *<username>*@132.187.14.67**

The master node is accessible only via the internal network of the University of Würzburg.

## For Beginners

GPUs are not available on the master node. To use GPUs on worker nodes, the JSON/YAML configuration must be run with **kubectl** command.

## Project Examples

To quickly verify that K8s is working properly in your namespace, run at the command line:

**cd /shared/ssd/examples/nn-dataset && kubectl delete -f quick-train.json || true && kubectl apply -f quick-train.json && sleep 10 && kubectl get pods --no-headers --field-selector=status.phase==Running -o custom-columns=":[metadata.name](http://metadata.name)" | grep nn-dataset-train-100 | head -n 1 | xargs kubectl logs -f**

You should see the training pipeline output within a few minutes. Delays may occur if resources are fully utilized or if a Docker image is downloading. If you encounter a delay or error, you can stop the command by pressing `Ctrl+C` and check the status of your pod with:

**kubectl get pod**

To explore the cluster, consider starting with a simple example such as training a classification model using the LEMUR project or fine-tuning a large language model with the NNGPT project:

**/shared/ssd/examples/lmur** and  **/shared/ssd/examples/nngpt**

To begin:

- Navigate to your shared home directory: **cd**  **/shared/ssd/home/*<user name>***
- Copy the example project to your shared home directory: **cp -r /shared/ssd/examples/lmur .**
- Set permissions for the Docker container to access the directory: **chmod -R 777 lmur**
- Move into directory 'lmur': **cd lmur**
- Update configuration files:  
  In all `*.json` files, replace `<user name>` with your actual username and `<user id>` with your numeric user ID (check with `id -u`).
- Download the project: **kubectl apply -f pull-nn-dataset.json**
- **⚠️ Common Issue:** Before running your script, make sure to delete an existing Kubernetes job with this name (e.g., **kubectl delete -f pull-nn-dataset.json**). Otherwise, an error will occur.
- You can monitor your pods with: **kubectl get pod** and **kubectl logs -f *<pod-id>***
- Start training the CNN model: **kubectl apply -f quick-train.json**
- To stop the training job: **kubectl delete -f quick-train.json**

You can follow a very similar set of steps to fine-tune large language models using the **NNGPT** project.

The provided project examples can be adapted for your own use by modifying them to load your projects from GitHub or run a local copy of code.

## Docker Images

You are welcome to use any Docker image of your choice, or you may opt for **AI Linux**, a custom image specifically built for CVL cluster. For usage details and examples, please visit:  <https://hub.docker.com/r/abrainone/ai-linux>

If dependencies are missing, create a container from a Docker image such as **abrainone/ai-linux**, install the required packages using `pip install <package name>`. Then, create a new image with: `docker commit <container name> <new image name>`. This custom image can be pushed to a registry for deployment on the computing cluster. As an alternative to installing missing lightweight dependency, you can add the installation command before the training command in the JSON config file. For example: `"pip install <package-name> && python ..."`

## K8s Registry

Feel free to create and push docker images to our registry **ws.ab** to make them accessible from your jobs.

This registry doesn't support permission management, so you are  responsible for keeping naming conventions. Please, name all your docker  images according to the scheme **ws.ab/*<user name>*/IMAGE-NAME**.

A naïve example of pushing a docker image to the registry **ws.ab**:

\---  
**docker pull abrainone/ai-linux**

**docker tag abrainone/ai-linux  ws.ab/*<user name>*/ai-linux**

**docker push ws.ab/*<user name>*/ai-linux**

**docker rmi abrainone/ai-linux  ws.ab/*<user name>*/ai-linux**

\---

Once pushed to the **ws.ab** registry, the image **ws.ab/*<user name>*/ai-linux** can be listed in your job by the new name.

To preserve disk space, remove all images you created from docker (**docker rmi *<image>***) after pushing them in the **ws.ab** registry. You can always put the image back.

For the shared by our team images we can use repository: **ws.ab/shared**.

Useful registry commands:

- List repositories: **curl -sS https://ws.ab/v2/\_catalog**

List images: **curl -sS https://ws.ab/v2/<repo>/tags/list (i.g. curl -sS [https://ws.ab/v2/a-t-test/cuda/tags/list](https://ws.ab/v2/a-t-test/cuda/tags/list))**

## Using the Home Directory

Due to the limited disk size on the CV cluster, only configuration files and programs that are important for connecting to the master nodes are stored in the home directory (**/home/*<username>***). All datasets are placed in the **/shared/hdd/data**, other files are kept on your local workstations or SSD directory (**/shared/ssd/home/*<user name>***).

## Shared Directories

Shared across all cluster nodes predefined directories for datasets, logs, examples and private projects, respectively:

- /shared/hdd/data
- /shared/ssd/logs
- /shared/ssd/examples
- /shared/ssd/home

## Command-Line Utilities

1. **datacp** copies dataset from NAS folder **/shared/hdd/data** to the local folders of worker nodes **/shared/local/data/*<user name>***. For more information, run the **datacp** command without parameters. Aside from the smallest datasets, it is critical to copy the data to local worker disks for efficient training of neural networks.
2. **idle** collects and displays information about available cluster resources.

## 

## Resource Management

- Worker nodes **W1 - W5** have 128 threads, which allows to request up to **16** CPUs per one GPU
- Worker nodes **W6 - W10** have only 32 threads, therefore requesting more than **8** CPUs per one GPU is **not recommended**.

## Q & A

- *I have problems with permission to access checkpoints and virtual environment files created by my docker container.*

  The problem can be solved by adding the **securityContext** into the YAML configuration of your job (for more details with correct YAML format, see the project examples mentioned above):

  ```
   securityContext:
     allowPrivilegeEscalation: false
     capabilities:
       drop:
       - ALL
     runAsUser: 1376
  ```

  were 1376 is id of the group assigned to all k8s users.
- *My job is running very slow.*

  This usually happens when a dataset is used from a remote NAS directory **/shared/hdd/data**. Copying your dataset using the **datacp** command line utility will solve this problem. For more information, run **datacp** without arguments.