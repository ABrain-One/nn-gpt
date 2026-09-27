Contact person: Florin Vasluianu [florin-alexandru.vasluianu@uni-wuerzburg.de](mailto:florin-alexandru.vasluianu@uni-wuerzburg.de)

# Julia2

CVL members can access the computing nodes from the Julia2 HPC center. This is the largest computational resource of the CVL, and every CVL member is expected to contribute in terms of administration and development.

These computers are provided to the scientific staff at the university level, and they are maintained and managed by the [Rechenzentrum](https://www.rz.uni-wuerzburg.de/).

The onboarding procedure its simple, and it requires usage of this [online form](https://www.rz.uni-wuerzburg.de/dienste/rzserver/high-performance-computing/benutzeraccount/) (for CVL members and guests). This requires you to have already an account activated in the integrated login system of the university (e.g. Uni employee/guest email account).

Here you need to select the group *Computer Vision Research (Leherstuhl für Informatik IV)*, and the same name for the research project.

When you fill in the form, a ticket is going to be automatically generated, and an username will be provided as the ticked is resolved. Storage for datasets and checkpoints will be available under /data/42-julia-hpc-rz-cv. 

For **Praktikum and Master Thesis students** working with advisors from CVL, the temporary access can be granted by Florin Vasluianu ([florin-alexandru.vasluianu@uni-wuerzburg.de](mailto:florin-alexandru.vasluianu@uni-wuerzburg.de)) or Dr. Dmytro Ignatov ([dmytro.ignatov@uni-wuerzburg.de](mailto:dmytro.ignatov@uni-wuerzburg.de)). 

Advisors are expected to send an email providing the student ID (s123xxxx) and the Uni Würzburg student email. Another email is expected after the final presentation, to clear  the student files under /home or /data/42-julia-hpc-ai-cv-students. 

# 

# Documentation

For general information, SLURM examples, and other guidelines, its is highly recommended that you go through the [Julia2 documentation](%20https://doku.hpc.uni-wuerzburg.de/).

Also, when you log in via SSH, another provided link is particularly useful. You can report bugs, errors, usage misbehavior, or ask for advice if you join the Julia2 Matrix chat.

For this, install an Matrix Client such as Element.io and join the following Matrix channel:

Matrix: #julia2-test:[matrix.albin-zehe.de](http://matrix.albin-zehe.de)

Requests to the system admins can be done through private messages to Aleks Paravac (username *watz*), as he is the primary system administrator.

The links for the documentation and the Matrix chat appear as a message every time you log in via SSH.

```
Linux julia2 6.1.0-26-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.1.112-1 (2024-09-30) x86_64
Last login: Thu Feb 13 20:26:04 2025 from 10.109.2.126

       _,met$$$$$gg.           flv45zx@julia2
    ,g$$$$$$$$$$$$$$$P.        --------------
  ,g$$P"         """Y$$.".     OS: Debian GNU/Linux bookworm 12.9 x86_64
 ,$$P'               `$$$.     Host: KVM/QEMU Standard PC (i440FX + PIIX, 1996) (pc-i440fx-9.0)
',$$P       ,ggs.     `$$b:    Kernel: Linux 6.1.0-26-amd64
`d$$'     ,$P"'   .    $$$     Uptime: 78 days, 13 hours, 16 mins
 $$P      d$'     ,    $$$P    Packages: 1478 (dpkg)
 $$:      $.   -    ,d$$'      Shell: bash 5.2.15
 $$;      Y$b._   _,d$P'       Cursor: Adwaita
 Y$$.    `.`"Y$$$$P"'          Terminal: /dev/pts/74
 `$$b      "-.__               CPU: AMD EPYC 7543 32-Core (16) @ 2,79 GHz
  `Y$$                         GPU: Unknown Device 1111 (VGA compatible)
   `Y$$.                       Memory: 1,99 GiB / 62,79 GiB (3%)
     `$$b.                     Swap: 2,31 GiB / 16,00 GiB (14%)
       `Y$$b.                  Disk (/): 31,14 GiB / 51,01 GiB (61%) - ext4
          `"Y$b._              Disk (/tmp): 162,84 GiB / 1,97 TiB (8%) - ext4
             `"""              Local IP (ens18): 10.106.223.223/24 *
                               Locale: en_US.UTF-8

                               ████████████████████████
                               ████████████████████████

Doku: https://doku.hpc.uni-wuerzburg.de/
Matrix: #julia2-test:matrix.albin-zehe.de
```

# Computers

In terms of GPU computation, Julia2 mainly consists of NVIDIA L40 and NVIDIA L40s GPU cards. Also,  16 NVIDIA H100 cards are available through the nodes *jnultra01* and *jnultra02*, as part of the partition *h100*.

On top of the GPU computation, running CPU-based simulations is also possible  through  the *small_cpu*  and  *large_cpu* partitions.

For more  information about partitions and nodes, consult the [Julia2 documentation](%20https://doku.hpc.uni-wuerzburg.de/), or run the command *sinfo*.

The output of the *sinfo* command will be similar to this:

::: info
PARTITION               AVAIL  TIMELIMIT  NODES  STATE NODELIST
small_cpu*                 up 2-00:00:00      1   mix- jn001
small_cpu*                 up 2-00:00:00      1   drng jn013
small_cpu*                 up 2-00:00:00     16    mix jn[002-010,012,014-019]
small_cpu*                 up 2-00:00:00      2  alloc jn[011,020]
large_cpu                  up 2-00:00:00     20   mix- jn[101-120]
standard                   up 1-00:00:00     23   mix- jn[001,101-120],jnfat[03,09]
standard                   up 1-00:00:00      1   drng jn013
standard                   up 1-00:00:00     16    mix jn[002-010,012,014-019]
standard                   up 1-00:00:00      5  alloc jn[011,020],jnfat[01-02,08]
h100                       up 1-00:00:00      2   mix- jnultra[01-02]
test                       up    1:00:00      1    mix jntest01
gpu_computervision         up 2-00:00:00      2    mix jnfat[04-05]
gpu_computervision         up 2-00:00:00      2   idle jnfat[06-07]
gpu_computervision_long    up 4-00:00:00      2    mix jnfat[04-05]
gpu_computervision_long    up 4-00:00:00      2   idle jnfat[06-07]
cpu_standard               up 2-00:00:00     11    mix jncpu[01,03-12]
cpu_standard               up 2-00:00:00      1  alloc jncpu02
cpu_long                   up 7-00:00:00      1  alloc jncpu14
cpu_long                   up 7-00:00:00      5   idle jncpu[13,15-18]

:::

This shows all the partitions, availability, and the nodes associated with each partition.

The last partition (e.g *gpu_computervision*)  is particularly useful. It consists of 4 computing nodes, each hosting 8x NVIDIA L40s cards.

This partition is accessible to the CVL  members, having scheduling priority over the other groups, and the computation is equally shared by the CVL users of Julia2.

# Useful commands and guidelines

The utilization degree of Julia2 can be checked though the [Julia2 Grafana Dashboard](https://10.106.223.186/d/bX7jn6dZk/slurm-dashboard?orgId=1&refresh=30s).

Here you can find information regarding the number of available CPU cores of GPU, and job related information.

More detailed information can be consulted through particular commands available in the SLURM toolkit.

For example, the following command shows the resources of each of the computation nodes and the resources currently in use.

Thus, you can determine which node can immediately accommodate your computing job if you request a set of resources (CPU cores, RAM, number of GPUs and the right partition name) which fits what is available at the moment.

```
scontrol -o --json show nodes | jq '.nodes[] | "\(.hostname) \(.tres) \(.tres_used)"'
```

The output of this command consists of two columns, in which the first shows the default resources, and the second shows the utilization at the moment.

::: info
"jn001 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=98,mem=192G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=33285996544"
"jn002 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=112,mem=256G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=3629247365120"
"jn003 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=86,mem=98G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=24696061952"
"jn004 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=100,mem=113G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=55834574848"
"jn005 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=76,mem=82G,gres/gpu=2,gres/gpu:l40=2,gres/tmp=19327352832"
"jn006 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=66,mem=18G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=18253611008"
"jn007 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=36,mem=50G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=2147483648"
"jn008 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=4,mem=48G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=1073741824"
"jn009 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 "
"jn010 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 "
"jn011 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 "
"jn012 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=4,mem=48G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=1073741824"
"jn013 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=4,mem=48G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=1073741824"
"jn014 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 "
"jn015 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=2,mem=32G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=17179869184"
"jn016 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=8,mem=96G,gres/gpu=2,gres/gpu:l40=2,gres/tmp=2147483648"
"jn017 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 "
"jn018 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=106,mem=160G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=238370684928"
"jn019 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=38,mem=96G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=19327352832"
"jn020 cpu=112,mem=480G,billing=112,gres/gpu=3,gres/tmp=3629247365120 cpu=36,mem=96G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=35433480192"
"jn101 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=64,mem=95G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=54760833024"
"jn102 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=6,mem=80G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=51539607552"
"jn103 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 "
"jn104 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=48,mem=64G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=215822106624"
"jn105 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=224,mem=256G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=3629247365120"
"jn106 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=132,mem=80G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=69793218560"
"jn107 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=130,mem=64G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=85899345920"
"jn108 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=96,mem=96G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=3221225472"
"jn109 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=36,mem=111G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=54760833024"
"jn110 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=4,mem=48G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=1073741824"
"jn111 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 "
"jn112 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=130,mem=48G,gres/gpu=3,gres/gpu:l40=3,gres/tmp=85899345920"
"jn113 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 "
"jn114 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=4,mem=48G,gres/gpu=2,gres/gpu:l40=2,gres/tmp=34359738368"
"jn115 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 "
"jn116 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=16,mem=16G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=5368709120"
"jn117 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=6,mem=64G,gres/gpu=2,gres/gpu:l40=2,gres/tmp=18253611008"
"jn118 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=2,mem=16G,gres/gpu=1,gres/gpu:l40=1,gres/tmp=17179869184"
"jn119 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 cpu=18,mem=48G,gres/gpu=2,gres/gpu:l40=2,gres/tmp=22548578304"
"jn120 cpu=224,mem=480G,billing=224,gres/gpu=3,gres/tmp=3629247365120 "
"jnfat01 cpu=128,mem=960G,billing=128,gres/gpu=8,gres/tmp=15354508083200 cpu=40,mem=144G,gres/gpu=4,gres/gpu:l40s=4,gres/tmp=12884901888"
"jnfat02 cpu=128,mem=960G,billing=128,gres/gpu=8,gres/tmp=15354508083200 cpu=86,mem=271G,gres/gpu=8,gres/gpu:l40s=8,gres/tmp=182536110080"
"jnfat03 cpu=128,mem=960G,billing=128,gres/gpu=8,gres/tmp=15354508083200 "
"jnfat04 cpu=192,mem=960G,billing=192,gres/gpu=8,gres/tmp=15354508083200 "
"jnfat05 cpu=192,mem=960G,billing=192,gres/gpu=8,gres/tmp=15354508083200 "
"jnfat06 cpu=192,mem=960G,billing=192,gres/gpu=8,gres/tmp=15354508083200 "
"jnfat07 cpu=192,mem=960G,billing=192,gres/gpu=8,gres/tmp=15354508083200 "
"jntest01 cpu=240,mem=480G,billing=240,gres/gpu=2,gres/tmp=3629247365120 "
"jnultra01 cpu=224,mem=1984G,billing=224,gres/gpu=8,gres/tmp=15354508083200 cpu=192,mem=128G,gres/gpu=6,gres/gpu:h100=6,gres/tmp=122406567936"
"jnultra02 cpu=224,mem=1984G,billing=224,gres/gpu=8,gres/tmp=15354508083200 cpu=144,mem=384G,gres/gpu=6,gres/gpu:h100=6,gres/tmp=96636764160"

:::

Another useful command is:

```
sinfo -N -r -l
```

The output will be similar to:

::: info
NODELIST   NODES          PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON

jn001          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn001          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn002          1           standard   allocated 128    2:32:2 491520  3461120      1   (null) none  
jn002          1         small_cpu*   allocated 128    2:32:2 491520  3461120      1   (null) none  
jn003          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn003          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn004          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn004          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn005          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn005          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn006          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn006          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn007          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn007          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn008          1           standard        idle 128    2:32:2 491520  3461120      1   (null) none  
jn008          1         small_cpu*        idle 128    2:32:2 491520  3461120      1   (null) none  
jn009          1           standard        idle 128    2:32:2 491520  3461120      1   (null) none  
jn009          1         small_cpu*        idle 128    2:32:2 491520  3461120      1   (null) none  
jn010          1           standard        idle 128    2:32:2 491520  3461120      1   (null) none  
jn010          1         small_cpu*        idle 128    2:32:2 491520  3461120      1   (null) none  
jn011          1           standard        idle 128    2:32:2 491520  3461120      1   (null) none  
jn011          1         small_cpu*        idle 128    2:32:2 491520  3461120      1   (null) none  
jn012          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn012          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn013          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn013          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn014          1           standard        idle 128    2:32:2 491520  3461120      1   (null) none  
jn014          1         small_cpu*        idle 128    2:32:2 491520  3461120      1   (null) none  
jn015          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn015          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn016          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn016          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn017          1           standard     drained 128    2:32:2 491520  3461120      1   (null) gpu-ecc-error  
jn017          1         small_cpu*     drained 128    2:32:2 491520  3461120      1   (null) gpu-ecc-error  
jn018          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn018          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn019          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn019          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn020          1           standard       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn020          1         small_cpu*       mixed 128    2:32:2 491520  3461120      1   (null) none  
jn101          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn101          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn102          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn102          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn103          1          large_cpu     drained 256    2:64:2 491520  3461120      1   (null) gpu-ecc-error  
jn103          1           standard     drained 256    2:64:2 491520  3461120      1   (null) gpu-ecc-error  
jn104          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn104          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn105          1          large_cpu   allocated 256    2:64:2 491520  3461120      1   (null) none  
jn105          1           standard   allocated 256    2:64:2 491520  3461120      1   (null) none  
jn106          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn106          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn107          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn107          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn108          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn108          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn109          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn109          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn110          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn110          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn111          1          large_cpu        idle 256    2:64:2 491520  3461120      1   (null) none  
jn111          1           standard        idle 256    2:64:2 491520  3461120      1   (null) none  
jn112          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn112          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn113          1          large_cpu        idle 256    2:64:2 491520  3461120      1   (null) none  
jn113          1           standard        idle 256    2:64:2 491520  3461120      1   (null) none  
jn114          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn114          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn115          1          large_cpu        idle 256    2:64:2 491520  3461120      1   (null) none  
jn115          1           standard        idle 256    2:64:2 491520  3461120      1   (null) none  
jn116          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn116          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn117          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn117          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn118          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn118          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn119          1          large_cpu       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn119          1           standard       mixed 256    2:64:2 491520  3461120      1   (null) none  
jn120          1          large_cpu        idle 256    2:64:2 491520  3461120      1   (null) none  
jn120          1           standard        idle 256    2:64:2 491520  3461120      1   (null) none  
jnfat01        1           standard    draining 128    2:64:1 983040  1464320      1   (null) gpu-ecc-error  
jnfat02        1           standard       mixed 128    2:64:1 983040  1464320      1   (null) none  
jnfat03        1           standard     drained 128    2:64:1 983040  1464320      1   (null) gpu-ecc-error  
jnfat04        1 gpu_computervision        idle 192    2:96:1 983040  1464320      1   (null) none  
jnfat05        1 gpu_computervision        idle 192    2:96:1 983040  1464320      1   (null) none  
jnfat06        1 gpu_computervision        idle 192    2:96:1 983040  1464320      1   (null) none  
jnfat07        1 gpu_computervision        idle 192    2:96:1 983040  1464320      1   (null) none  
jntest01       1               test        idle 256    2:64:2 491520  3461120      1   (null) none  
jnultra01      1               h100       mixed 224    2:56:2 203161  1464320      1   (null) none  
jnultra02      1               h100       mixed 224    2:56:2 203161  1464320      1   (null) none

:::

This will be particularly useful to determine which nodes currently experience errors, and will have to be rebooted in future.

A node appearing with the state *draining* will not get any new job requests from the scheduler, and after all the jobs that are currently run on it will be finished, it can be safely rebooted by the system admins.

Based on these two commands, you can identify if the requested resources can be easily accomodated by any node, such that the waitingtime in the job queue is minimal.

#  Checking Julia2 group members

The CVL members of Julia2 will be allocated in the groups *42-julia-hpc-rz-cv* (CVL members and guests) or *42-julia-hpc-ai-cv-students* (Praktikum students/Master Thesis students/other collaborators). 

Checking the persons from a group can be done with a command similar to the next example, which lists the accounts of the students affiliated with CVL. Changing the group name to  *42-julia-hpc-rz-cv* will select the CVL staff with Julia2 access.  

```
for i in $(getent group | grep 42-julia-hpc-ai-cv-students | cut -d ":" -f 4 | xargs -d ',' -n1); do getent passwd | grep $i; done;
```

# Frugal and responsible usage of the Julia2 resources

Julia2 resources are shared within the group, so responsible usage is crucial for the activity of CVL. Thus, the following command shows the utilization degree of the Julia2 resources for all its users. 

For example, the following command should output a list of all users active on Julia2 sttarting 01/08/2026 until the moment the command is run, with an heuristic function quantifying the ammount of used resources. 

If it happens for you to have used more resources than the average of CVL users, preoceed to scale down your experiments such that fair resource access is achieved, without other colleagues being blocked.

 In case of urgent deadlines coordinate with your Postdocs or advisors to prioritize the necessary experiements and balance the resource needs with the available budget.  

```
sreport cluster AccountUtilizationByUser start=2026-08-1 end=now 
```

# Responsible GPU usage

Given the fact that the group budget is equally shared by its memebers, its crucial that the GPU hours are spent as wisely as possible by all CVL members: 

Please consider the following factors:

- in case of new experiments, consider using the *test* partition to make sure your code works;
- perform periodic checkpoints/results saving;
- do not follow tutorials from baseline models blindly. Try to adapt the software to the Julia2 requirements whenever needed;
- if the size of the dataset used for experimentation is low, consider allocating space on */tmp*, then copy your dataset from */data* to */tmp*, and then load it from there for training/validation;
- balance input sizes/batch sizes based on the total VRAM available on the Julia2 GPUs (48 GB for L40/L40s and 80GB for H100). For this Julia2 allows interactive access to the container hosting the running job.

```
# identify your jobs
squeue --me
# suppose your job ID is 3332320
srun --pty --overlap --jobid 3332320 bash
# now your bash would switch for the one on the node hosting the job
# so here you can run your own monitoring command
watch -n1 nvidia-smi
# this will update every 1 sec. showing GPU dtails
# balance your hyperparameters for the best possible usage statistics

# now, to exit from the 'watch' CLI press CTRL+C
# to exit the node CLI simply use the 'exit' command
exit
```

# Responsible storage usage

Julia 2 is build around two main partitions: */home* and */data*. The */home* partition should be used for virtual environments, software, and your projects codes. Since */home*  has a quota of \~200GB per user, you can afford to host there some small datasets if needed (would train faster) and most of your checkpoints.

The */data* partition is particularly slowed down by large collections of very small files. Consider using data collections (e.g. image grids) for any large collections of small files stored under */data.* Do NOT save Python *virtualenvs* on */data* under any circumstances. They consist of at least 200k of very small files, leading to extreme fragmentation. 

REMEMBER TO CLEAN unnecessary files regularly, from both */home* and */data*. 

#  CVL own queues

The ‘*sinfo*’ command will output two specifiec queues, namely ‘*gpu_cumputervision*‘ and ‘*gpu_computervision_long*‘. The jobs on these queues will run exclusively on CVL owned hardware. Since Sept 2026 the CVL hardware was made available to the other Julia2 users, on the condition that jobs submitted by CVL members will cause the preemption of any job submitted to these queues by other users. Therefore, even if the partition is currently fully occupied, CVL members can still submit their jobs, and after the expiration of a preeption time of 1h the jobs of CVL memebers will automatically start. 

Thus, the procedure could be something similar to the following proposal:

```
# Check the nodes associated to the gpu_computervision queues
scontrol -o --json show nodes | jq '.nodes[] | "\(.hostname) \(.tres) \(.tres_used)"' | grep jnfat
# check the resources available on jnfat04, jnfat05, jnfat06, jnfat07

# then check the jobs in the gpu_computervision queues
squeue | grep gpu_comp

# to check the users associated with CVL (employees, guests, students)
for i in $(getent group | grep 42-julia-hpc-rz-cv | cut -d ":" -f 4 | xargs -d ',' -n1); do getent passwd | grep $i; done;
for i in $(getent group | grep 42-julia-hpc-ai-cv-students | cut -d ":" -f 4 | xargs -d ',' -n1); do getent passwd | grep $i; done;
# therefore the jobs submitted by users outside of this list are subject to the preemption rules and the resources associated to them can be considered available
```

# To Be Used Wisely

Since Sept 2026 the *squeue* command will list the whole list of Julia2 jobs in the cluster at the moment, regardless the state of the jobs. Therefore, it is easy to find the users running jobs of the *gpu_computervision* partitions. For example:

```
# list all the jobs running on the jnfat nodes
squeue | grep jnfat
# list the node allocation per pertition 
sinfo -N -r -l | grep gpu_computervision
# get the usernames associated with the jobs running on our hardware

# to check the group of a particular user you can replace xxxxxxx with the particular username in the following command
sacctmgr show associations user=xxxxxxx
```

# Job submission guidelines

 1. Julia2 is the main computational resource of CVL. Prioritize Julia2 environments over K8s or Vingilot setups.
 2. Migrating from K8s to julia2 is not difficult. In theory, the creation of a simple <paragraph></paragraph>working Python virtual environment (check virtualenv) it's similar to the Docker image definition in K8s.
 3. Keep in mind the fact that Julia2 is a shared resource and the amount of available resources is limited.
 4. Testing your code is available through the *test* partition. You can run a toy example there to check if your job definition is right before submitting to the *standard* or *h100* partitions.
 5. Participate in the discussions regarding any errors you can observe, and feel free to propose improvements based on your experience. Propose there possible improvements in the Matrix chat group and support your ideas in front of the community.
 6. As a member of CVL, access to a separate partition is available. However, these computers are shared will all your colleagues. Please do not prioritize the *gpu_computervision*  partition over the *standard* or *h100* partitions.

    The *gpu_computervision* is intented to be used as a buffer when the other public partitions are fully occupied (or there is no node that can accomodate you job in the public partitions).
 7. When submitting a job, check the state of the nodes, the free resources, and avoid requesting or blocking too many resources. The amount of used resources compared to your requests can be analyzed by the sysadmins and bad resource management will be reported.
 8. CPU and RAM resource requests need to be proportional to the available resources, given the requested GPUs ratio.
 9. When the utilization is high, waiting in the job queue is normal. Continue submitting your jobs such that your job starts immediately after your resource requests can be met byu the system.
10. Bad behavior and inefficient resource management will be penalized by decreasing user priorities and temporary access restrictions.  Be nice to your colleagues who need *gpu_computervision* resources. If you are blocking more than your defaut share (32 GPUS / number_of_users), stop some of your jobs in case somebody urgently needs resources.