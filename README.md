"# DreamScape" 
# Inspiration

In the early 2010s, the endeavor to "read minds" by generating images of what a person was viewing through EEG scans was predominantly spearheaded by convolutional neural networks and other foundational computer vision techniques. Although promising, these methods faced significant challenges in accurately decoding and replicating intricate visuals. However, with the recent rise of transformer-based models and sophisticated neural architectures, many of these initial challenges have been overcome. Armed with these advanced tools, we recognized a chance to revisit and rejuvenate this field. Beyond the technological intrigue, there's a profound purpose: by converting the dreams of dementia patients into visual narratives, we aspire to make substantial advances in decoding the mysteries of Alzheimer's and associated cognitive disorders.

# What it does

**DreamScape** represents a sophisticated blend of neurology and AI. The process begins with high-resolution EEG scans that record the intricate brainwave patterns exhibited during dreams. These patterns are subsequently inputted into a deep learning model, specially trained using convolutional layers, which translates the EEG signals into basic images and relevant textual descriptions. To mold this data into a cohesive narrative, we deploy advanced natural language processing models, particularly transformer architectures from the GPT series. The final phase involves the generation of a detailed visual portrayal using Generative Adversarial Networks (GANs), crafting lifelike scenes inspired by the earlier narrative outputs.

The model in question produces an image for each 0.5 seconds of EEG data, allowing us to convert any visual experience (including dreams) into a sequence of images.  EEG data can be easily and nonintrusively collect with a device such as Muse. Our tool then uses Generative AI tools to stitch the information from each of these images into a video story that hopefully replicates the story in your dream. From frequent snapshots of the story, we hope to fill in the gaps. 

# How we built it

The EEG data are sourced from reputable research journals. Our machine learning foundation leverages **TensorFlow** and **Hugging Face's Transformers** library, chosen for their synergy with intricate neural architectures. Additionally, **OpenAI's GPT API** bolsters our narrative generation process, with its pre-trained models minimizing our training overhead. For the visual narrative, ensuring continuous and coherent visual output, we employ a modified version of stable diffusion techniques. This guarantees visuals that flow seamlessly, much like a dream. Our web application interface, tailored for both researchers and end-users, utilizes the capabilities of **Next.js** and **React** for dynamic UI components and **Flask** as a nimble backend server for data processing and model interactions.

#Video_Models

The Video Models folder will be uploaded in the future. The model used is a varient of Deforum Stable Diffusion, which is modified to produce a series of images that flow between prompts, and the prompts are derived from the images produced from EEG data. In the future, we hope to avoid this text intermediary and go directly from image to video.
