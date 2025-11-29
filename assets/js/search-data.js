// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/GCMDP-blogpost/";
    },
  },{id: "nav-about",
          title: "about",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/GCMDP-blogpost/about/";
          },
        },{id: "nav-call-for-blogposts",
          title: "call for blogposts",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/GCMDP-blogpost/call/";
          },
        },{id: "nav-submitting",
          title: "submitting",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/GCMDP-blogpost/submitting/";
          },
        },{id: "nav-reviewing",
          title: "reviewing",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/GCMDP-blogpost/reviewing/";
          },
        },{id: "dropdown-lt-strong-gt-2026-lt-strong-gt",
              title: "&lt;strong&gt;2026&lt;/strong&gt;",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2026/";
              },
            },{id: "dropdown-2025",
              title: "2025",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2025/";
              },
            },{id: "dropdown-2024",
              title: "2024",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2024/";
              },
            },{id: "dropdown-2023",
              title: "2023",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2023/";
              },
            },{id: "dropdown-2022",
              title: "2022",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blog-track.github.io/home/";
              },
            },{id: "post-reward-maximization-using-goal-conditioned-rl",
        
          title: "Reward maximization using Goal-conditioned RL",
        
        description: "TODO - this is abstract",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/GCMDP/";
          
        },
      },{id: "post-wait-do-we-need-to-wait-revisiting-budget-forcing-for-sequential-test-time-scaling",
        
          title: "Wait, Do We Need to Wait? Revisiting Budget Forcing for Sequential Test-Time Scaling...",
        
        description: "This blog revisits budget forcing, a sequential test-time scaling technique for reasoning models by controlling when it continues thinking versus when it must answer. We evaluate how well the method transfers across model types, including non-reasoning models, and whether alternative keywords work. We provide practical guidelines for using the technique.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/wait-do-we-need-to-wait/";
          
        },
      },{id: "post-symbolism-outside-connectionism-inside-the-trend-of-fusing-llms-and-automatic-programs-with-symbolic-intermediate-representations",
        
          title: "Symbolism Outside, Connectionism Inside: The Trend of Fusing LLMs and Automatic Programs with...",
        
        description: "This blog introduces the trend of fusing Large Language Models (LLMs) with external symbolic programs as a new paradigm in modern and future artificial intelligence (AI). This paradigm regards LLM output as a symbolic intermediate representation (IR), which is interpreted and executed by external symbolic programs to achieve the desired behavior. We firstly review and summarize the diverse applications of this paradigm. Then we introduce the more possible usages of this paradigm, from synthesizing grounded training data to composing modular systems of specialized neural networks. Finally, we introduce the frontier of this approach: applying formal methods to automatically verify the LLM&#39;s internal reasoning processes and outputs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/symbolic-connect/";
          
        },
      },{id: "post-getting-sac-to-work-on-a-massive-parallel-simulator-an-rl-journey-with-off-policy-algorithms",
        
          title: "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With...",
        
        description: "This post details how to get the Soft-Actor Critic (SAC) and other off-policy reinforcement learning algorithms to work on massively parallel simulators (e.g., Isaac Sim with thousands of robots simulated in parallel). In addition to tuning SAC for speed, the post also explores why SAC fails where PPO succeeds, highlighting a common problem in task design that many codebases share.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/sac-massive-sim/";
          
        },
      },{id: "post-pushing-meta-continual-learning-algorithms-to-the-limit",
        
          title: "Pushing Meta-Continual Learning Algorithms to the Limit",
        
        description: "Meta-continual learning algorithms should be able to handle tasks with extended data streams compared to the traditional deep learning setting. These algorithms have not been applied to settings with extreme data streams, such as classification tasks with 1,000 classes, nor have they been compared to traditional continual learning algorithms. We compare meta-continual learning to continual learning and we find that meta-continual learning scales better than continual learning.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/pushing-meta-cl-methods/";
          
        },
      },{id: "post-language-as-a-window-into-the-mind-how-nlp-and-llms-advance-human-sciences",
        
          title: "Language as a Window Into the Mind: How NLP and LLMs Advance Human...",
        
        description: "Can NLP predict heroin-addiction outcomes, uncover suicide risk, or simulate (and even influence) brain activity? Could LLMs one day contribute to research worthy of a Nobel Prize for advancing our understanding of human behavior? And what role do NLP scientists play in shaping that possibility? This post explores these questions, arguing that language technologies are not just tools that support scientific work (like literature search agents, writing tools, or coding assistants), but that by treating language as a window into the human mind, NLP and LLMs can actively help researchers uncover mechanisms of human behavior, cognition, and brain function.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/nlp-for-human-sciences/";
          
        },
      },{id: "post-model-misspecification-in-simulation-based-inference-recent-advances-and-open-challenges",
        
          title: "Model Misspecification in Simulation-Based Inference - Recent Advances and Open Challenges",
        
        description: "Model misspecification is a critical challenge in simulation-based inference (SBI), particularly in neural SBI methods that use simulated data to train flexible neural density estimators. These methods typically assume that simulators faithfully represent the true data-generating process, an assumption that is often violated in practice. Resulting discrepancies can make observed data effectively out-of-distribution relative to the simulations, leading to biased posterior distributions and misleading uncertainty quantification. This post reviews recent work on model misspecification in neural SBI, covering formal definitions, methods for detection and mitigation, and their underlying assumptions. It also discusses practical implications for SBI workflows and outlines open challenges for developing robust SBI methods that remain reliable in realistic, imperfectly specified applications.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/model-misspecification-in-sbi/";
          
        },
      },{id: "post-do-language-models-really-learn-to-mislead-humans-via-rlhf",
        
          title: "Do Language Models Really Learn to Mislead Humans via RLHF?",
        
        description: "This post details an investigation of claims in Language Models Learn to Mislead Humans Via RLHF (ICLR 2025) that RLHF may unintentionally lead LLM agents to mislead humans (U-Sophistry). We found that the misleading behavior in the paper is the result of an unrealistic experimental setup and not of U-Sophistry, and can therefore be categorized as intentional misleading (I-Sophistry).",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/mislead-lm/";
          
        },
      },{id: "post-on-the-measure-of-a-model-from-intelligence-to-generality",
        
          title: "On the Measure of a Model - From Intelligence to Generality",
        
        description: "Benchmarks like ARC, Raven-style puzzles, and the Blackbird Task are often treated as measures of LLM intelligence. But intelligence is a moving target—hard to define and even harder to link to what we actually need models to do, like answer questions, summarize text, or write code. Optimizing for these abstract tests can pull evaluation away from real-world usefulness. We argue for a shift from chasing intelligence to measuring generality. This reframes how progress in AI should be assessed and proposes generality as a more stable foundation for evaluating capability across diverse and evolving tasks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/measuregen/";
          
        },
      },{id: "post-research-directions-in-multimodal-chain-of-thought-mcot-with-sketching",
        
          title: "Research Directions in Multimodal Chain-of-Thought (MCoT) with Sketching",
        
        description: "This article explores adding sketching to Multimodal Chain-of-Thought (MCoT)reasoning to enhance AI capabilities. It reviews past methods, identifies key gaps such as the lack of sketch-rationale datasets, and proposes advancing the field through targeted data collection, unified multimodal models, and reinforcement learning. Ethical considerations include mitigating cultural bias and visual misrepresentation in generated sketches.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/mcot-sketching/";
          
        },
      },{id: "post-from-reinforce-to-dr-grpo-a-unified-perspective-on-llm-post-training",
        
          title: "From REINFORCE to Dr. GRPO: A Unified Perspective on LLM Post-Training",
        
        description: "Recently, many reinforcement learning (RL) algorithms have been applied to improve the post-training of large language models (LLMs). In this article, we aim to provide a unified perspective on the objectives of these RL algorithms, exploring how they relate to each other through the Policy Gradient Theorem — the fundamental theorem of policy gradient methods.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/llm-post-training/";
          
        },
      },{id: "post-are-dilemmas-and-conflicts-in-llm-alignment-solvable-a-view-from-priority-graph",
        
          title: "Are Dilemmas and Conflicts in LLM Alignment Solvable? A View from Priority Graph...",
        
        description: "As Large Language Models (LLMs) become more powerful and autonomous, they increasingly face conflicts and dilemmas in many scenarios. We first summarize and taxonomize these diverse conflicts. Then, we model the LLM&#39;s preferences to make different choices as a priority graph, where instructions and values are nodes, and the edges represent context-specific priorities determined by the model&#39;s output distribution. This graph reveals that a unified stable LLM alignment is very challenging, because the graph is not static in different contexts. Besides, it also reveals a potential vulnerability: priority hacking, where adversaries can craft deceptive contexts to manipulate the graph and bypass safety alignments. To counter this, we propose a runtime verification mechanism, enabling LLMs to query external sources to ground their context and resist manipulation. While this approach enhances robustness, we also acknowledge that many ethical and value dilemmas are philosophically irreducible, posing an open challenge for the future of AI alignment.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/llm-conflicts/";
          
        },
      },{id: "post-justrl-scaling-a-1-5b-llm-with-a-simple-rl-recipe",
        
          title: "JustRL: Scaling a 1.5B LLM with a Simple RL Recipe",
        
        description: "Training small reasoning models with RL has become a race toward complexity, using multi-stage pipelines, dynamic schedules, and curriculum learning. We ask whether this complexity necessary? We show that JustRL, a simple recipe with fixed hyperparameters, achieves state-of-the-art performance on two different 1.5B base models (54.5% and 64.3% across 9 math benchmarks) while using 2× less compute than sophisticated approaches. The same hyperparameters transfer across both models without tuning, and training remains stable over thousands of steps without intervention. This suggests the field may be adding complexity to solve problems that disappear with a stable, scaled-up baseline.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/justrl/";
          
        },
      },{id: "post-introduction-to-stochastic-interpolants",
        
          title: "Introduction to Stochastic Interpolants",
        
        description: "Prominent generative modeling frameworks such as Flow Matching and score-based Diffusion Models establish a smooth transformation between a Gaussian distribution and a data distribution. In this blog post, we provide an introduction to the more general framework of Stochastic Interpolants, which allows one to flexibly interpolate between any two distributions and learn a velocity field to transform samples from one into samples of the other. No prior knowledge of generative models is required for this introduction.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/introduction-to-stochastic-interpolants/";
          
        },
      },{id: "post-evalcards-for-standardized-evaluation-reporting",
        
          title: "EvalCards for Standardized Evaluation Reporting",
        
        description: "In the age of rapidly released LLMs, evaluation reporting is fragmented, inconsistent, and often misleading. We surveyed the landscape and found three critical crises—reproducibility, accessibility, and governance—that Model Cards alone can&#39;t solve. Our solution? EvalCards-- lightweight, standardized evaluation summaries that are easy to write, easy to understand, and impossible to miss. EvalCards are designed to enhance transparency for both researchers and practitioners while providing a practical foundation to meet emerging governance requirements.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/evalcards/";
          
        },
      },{id: "post-ai-fundamentals-valuing-ai-agents-amp-data-assets",
        
          title: "AI Fundamentals: Valuing AI Agents &amp; Data Assets",
        
        description: "Large Language Model (LLM) agents now read the world through managed-context pipelines, write to it via tool-calling APIs, and continuously re-wire themselves with fresh experience. Stakeholders therefore need a Generally Accepted Accounting Principles (GAAP) compatible method to price both (i) the agent&#39;s labour-like output and (ii) the data traces that fuel learning. We formalise a single unifying metric - agent Economic Value (AEV)- and demonstrate, using evidence from millions of real conversations, $1M of freelance software contracts, and 48k physician rubric points in healthcare, that these metrics are measurable today. We then extend the template to reinforcement-learning regimes in which grounded rewards equal cash flows. Lastly, we propose a financial settlement layer, which transforms the agent from a passive software user into an active economic participant.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/economic-agents/";
          
        },
      },{id: "post-sample-blog-post",
        
          title: "Sample Blog Post",
        
        description: "Your blog post&#39;s abstract. Please add your abstract or summary here and not in the main body of your text. Do not include math/latex or hyperlinks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/distill-example/";
          
        },
      },{id: "post-from-u-nets-to-dits-the-architectural-evolution-of-text-to-image-diffusion-models-2021-2025",
        
          title: "From U-Nets to DiTs: The Architectural Evolution of Text-to-Image Diffusion Models (2021–2025)",
        
        description: "A comprehensive analysis of how diffusion model architectures evolved from U-Net backbones to Diffusion Transformers, transforming text-to-image generation capabilities.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/diffusion-architecture-evolution/";
          
        },
      },{id: "post-defining-and-quantifying-compositional-structure",
        
          title: "Defining and quantifying compositional structure",
        
        description: "Compositionality is thought to be crucial in human cognition and AI, but we lack a scientific understanding of what it is. What kind of data is compositionally structured? Can we mathematically quantify the amount and character of compositional structure? This blog post introduces a novel approach for doing so, building off of existing tools from algorithmic information theory that formalize notions of complexity and structure. The mathematical definition of compositionality that we&#39;ll come to is rigorous, precise, and general, and the hope is that it can inspire novel research directions in AI for uncovering compositional structure in natural data.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/compositionality/";
          
        },
      },{id: "post-chunktabpfn-training-free-long-context",
        
          title: "ChunkTabPFN: Training-free Long Context",
        
        description: "Tabular foundation models struggle with large datasets due to the quadratic attention. While methods like FlashAttention promise scalability, practical challenges persist in their application to tabular foundation models. Our work resolves these hurdles, enabling efficient attention, and reveals that contrary to the eariler reports, TabPFN&#39;s performance improves with larger contexts, highlighting its inherent robustness and minimal fine-tuning needs when scaling to complex, long datasets from the TabArena benchmark.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/chunked-tabpfn/";
          
        },
      },{id: "post-budget-alignment-making-models-reason-in-the-user-39-s-language",
        
          title: "Budget Alignment: Making Models Reason in the User&#39;s Language",
        
        description: "We explore a two step multilingual alignment recipe for large language models to keep reasoning and answers in the user language while preserving accuracy.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/budget-alignment/";
          
        },
      },{id: "post-gcmdp",
        
          title: "Gcmdp",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/GCMDP-blogpost/blog/2026/GCMDP/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/GCMDP-blogpost/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/GCMDP-blogpost/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/GCMDP-blogpost/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/GCMDP-blogpost/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
