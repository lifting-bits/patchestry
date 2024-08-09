# MLIR-based binary patching framework

Patchestry aims to make the same impact to binary patching as compilers and high
level languages did to early software development. Its main goal is to enable
developers without extensive knowledge of the deployment platform of the binary
to patch the binary. To do this, the developer has to be confident that what
they're patching is functionally equivalent to what is deployed and also that
the patch they write will integrate into the deployed binary without issue.

Patchestry leverages MLIR as the foundational technology instead of LLVM IR.
MLIR is an emerging compiler development technology allowing for the
specification, transformation, and mixing of IR dialects. The MLIR approach has
significant industry momentum, and has been adopted by companies such as Google
(in TensorFlow) and Meta (ClangIR). With MLIR, the decompilation process could
be stratified into a Tower of IRs (IR dialects). Each IR represents the same
program, but at a different level of abstraction.

MLIR brings a notable advantage by enabling the creation of representations to
streamline communication between diverse state-of-the-art tools. For instance,
one can create an MLIR dialect specifically for P-Code (a program representation
utilized by Ghidra) to optimize integration with the Ghidra decompiler.
Alternatively, an LLVM IR dialect can be employed to compile back to the
executable, and MLIR can support LLVM-based contract validation through a
symbolic executor such as KLEE. Moreover, MLIR provides flexibility to
devise our own dialects for representing contracts in specialized logic, such as
SMT. Finally, our [high-level dialect](https://trailofbits.github.io/vast/),
developed under DARPA V-SPELLS, captures the intricacies of full-featured C. Our
compiler stacks empower us to compile C into any of the previously mentioned
representations, promoting seamless interconnection between them.

## Technical Rationale

Our recent experience on AMP, as well as our performance on other DARPA binary
analysis programs (PACE, CFAR, LOGAN, CGC, Cyber Fast Track), have led us to
four guiding principles that we believe patching solutions for legacy software
must follow in order to be successful.

1. __Fully automated approaches are doomed to failure.__ In general, the process of
decompilation is an inherently intractable problem. However, developers are
often capable of distinguishing between decompilation outcomes deemed 'good' or
'bad', but encoding that kind of heuristic logic into a system invariably yields
unpredictability and unsoundness.  Hence, we assert that the involvement of
semi-skilled or skilled human developers is essential in the process. The
best-case scenario is that a developer can use an existing source code patch as
a guide. Given this patch, they can locate the corresponding vulnerable machine
code within a binary using binary function symbol names. The worst-case scenario
involves the ad hoc application of tools (e.g. BinDiff, BSim) and reverse
engineering skills to an opaque binary blob that is without symbols or debugging
information.

2. __Developers must be able to leverage pre-existing software development experience__
and not have to concern themselves with low level details. That is, they should
be able to operate as if the original source code and build process/environments
were available, and not be expected to have expert knowledge of every machine
code language that may be encountered.


3. __From-scratch development efforts do not scale.__ As much as possible,
pre-existing tooling that already handles the inherent scalability challenges in
(de)compiling code for such a wide variety of platforms should be leveraged. For
example the Ghidra decompiler can decompile over 100 machine code languages to
C, and the Clang compiler can generate machine code for over 20 machine code
languages. Rolling new solutions from scratch is impractical.

4. __There is no one-size-fits-all way of representing code.__ A “complete”
solution to machine code decompilation only exists at the end of a long tail of
special cases. Patchestry aims to provide decompilation to a familiar, C-like
language. Patchestry will not, however, decompile to C or a specific ISO dialect
thereof because some machine code constructs have no equivalents in C, while
others are only loosely equivalent given non-conforming dialect extensions.

## Project Goals

Patchestry accomplishes its goals by integrating various innovative concepts
guided by the four principles described in [technical rationale](#technical-rationale):

### Unified Tooling Integration

Guided by the third principle—recognizing the limitations of from-scratch
development efforts—Patchestry seamlessly integrates existing tooling for
decompilation and recompilation in the binary patching process. Patchestry
advocates a unified tooling integration approach using an MLIR Tower of
Intermediate Representations (IRs) as a mediator between tools. This strategy
enables the incorporation of cutting-edge (de)compiler tools into a cohesive
system, allowing the utilization of specialized tools for each task and ensuring
effectiveness and optimal outcomes across all desired functionalities.

### Incremental Decompilation

Patchestry’s innovative approach involves leveraging multiple program
representations simultaneously across various layers of the Tower of IRs. While
state-of-the-art decompilers already offer diverse representations, what sets
the Tower of IRs apart is its capability to create custom user-defined
abstractions (layers) while preserving relationships between these layers. This
modular approach facilitates seamless incremental decompilation and
recompilation processes. This is crucial for effortlessly devising specific
abstractions tailored to unique platforms.

### Unifying Representations for Contracts, Patches, and Software

The Tower of IRs also aligns with the fourth guiding principle: There is no
one-size-fits-all way of representing code.  Maintaining multiple
representations simultaneously in the Tower of IRs allows us to establish
meaningful relationships between them and innovate in how we connect tools and
conduct analyses. Additionally, this approach allows us to consolidate all
necessary components for patching within the same representation: patch
description, contract description and the software. This unified strategy
streamlines tooling for analysis and facilitates the recompilation of patched
software, resulting in a single artifact that can undergo desired formal
analyses, such as LLVM-based analysis.


### Declarative Patching and Contracts Description

To address our second guiding principle, which emphasizes the importance of
allowing developers to leverage their existing software development experience,
we mandate that all interactions with patching occur in a language commonly
understood by developers. Specifically, a C-like language. To facilitate this,
we propose a declarative library designed for describing patches, their
application. Following the same principle, Patchestry introduces contracts in
C-like DSL. These contracts serve as constraints guiding both decompilation and
recompilation, and they must hold at all relevant steps of each process.

## Decompilation Workflow

![Patchestry ls](img/patchestry-workflow.svg)

Patchestry’s technical approach is designed to enable the following seven-step workflow:

1. A developer is tasked with patching a vulnerability in a program binary
running on a device. How the user acquires a copy of the binary (e.g.
downloaded from a vendor’s website, extracted from a network capture, extracted
directly from a device over serial port or JTAG, etc.) is not part of the project.

2. The developer loads the binary into the open-source Ghidra interactive
decompiler. Developers will be enabled to leverage Ghidra’s features and plugins
to locate the function(s) to patch, though previous binary analysis expertise is
not required. We anticipate that developers will apply tools such as BinDiff or
BSim, rely on symbol names or debug information, or apply reverse engineering
techniques.

The Patchestry workflow includes Ghidra because it is open-source and actively
maintained by the National Security Agency and because it supports a wide
variety of binary file formats (ELF, COFF, PE, etc.) and machine code languages
used by medical devices. Ghidra also implements a battery of heuristics that act
as good first guesses as to the locations and references between functions and
data in the binary. Although perfect identification/recovery of functions, data,
and data types in a binary is intractable, the value of interactivity in Ghidra
is that the human developer can fix incorrect conclusions drawn by the
decompiler’s heuristics.

There are two reasons why Patchestry’s workflow does not allow the developer to
modify Ghidra’s decompilation output and then re-compile that into a patchable
representation. First, Ghidra’s decompilation is not guaranteed to be
syntactically correct or compilable. This can be mitigated through developer
effort; however, the level of effort increases with the complexity of and number
of references in the target function(s). Second, Ghidra’s heuristic
decompilation pipeline has been proven to be unfaithful with respect to the
execution semantics of the machine code. In the worst case, this could result in
a developer inadvertently introducing new vulnerabilities into the program
during the patching process.

Despite Ghidra’s decompilation not being precise enough for recompilation, our
experience from AMP tells us that Ghidra’s decompilation is good enough to be a
productivity multiplier for developers trying to locate functions that need
patching.

Moreover, the modular design of Patchestry affords the flexibility to seamlessly
integrate more formally rigorous decompilers and their representations in the
future, as their capabilities align with our technical requirements. Currently,
the majority of existing tools are predominantly of a research-oriented nature,
often concentrating on x86 architecture or even just its subset, which is
not sufficient for the diverse nature of software.

3. After locating the relevant function(s) in Ghidra, the Patchestry plugin will
present the developer with an editable decompilation of the target function(s).
Patchestry’s decompilations will be sound and precise with respect to the
available information in Ghidra’s analysis database. Regardless of how small the
patch size could be, Patchestry will always formulate the problem at the
function granularity. There are theoretical and pragmatic reasons why
Patchestry’s minimum patch recompilation granularity is function-at-a-time.

From a theoretical standpoint, function granularity patches enable Patchestry to
leverage stronger guarantees about the application binary interface (ABI). It is
only at the entry and exit points of a compiled function that higher-level,
human-readable types can be reliably mapped to low level machine locations
(registers, memory).

Patchestry leverages the open-source Clang compiler, which can already
target relevant platforms. A restriction in compilers like Clang that
nonetheless favors our approach is that functions are the smallest compilable
unit of code. Our task in Patchestry is thus to convert code for recompilation
into LLVM IR functions, which Clang can convert to machine code.

4. The developer edits the decompiled function(s), enacting the necessary
changes to patch the vulnerability in the decompiled code. Patchestry’s highest
level decompiled code (C-like) will look approximately similar, regardless of
the platform/architecture of the medical device software. This will help improve
developer productivity. Moreover, the meta-patch library will allow the
developer to automate the patching process.

At this stage, the binary-level patch has not yet been formulated. What
particular changes are needed to patch a given vulnerability are beyond the
scope of the project and require an external tool. Patchestry will, however,
provide a library of “patch intrinsics” such as “add bounds check.” These will
be formulated as templates of meta-patches.

A developer can make near arbitrary changes within the body of the decompiled
code (e.g. add, remove or replace its portions). Although Patchestry aims to
provide verifiable guarantees about feature- and bug-compatibility of its
decompilation with respect to the Ghidra database, absent contracts or
specifications about the intended behavior of the code, Patchestry cannot make
guarantees about the correctness of the edited decompilation. That is,
Patchestry cannot prevent a developer from introducing new flaws into the
binary, nor can it guarantee that a patch comprehensively fixes the root cause
of the vulnerability.

To mitigate the problem of developer- or decompiler-introduced emergent
behaviors, Patchestry will allow developers to leverage model- and
contract-based software verification techniques. These techniques are normally
challenging to apply to lifting/decompilation due to a lack of end-to-end
visibility into the lifting process; usually the techniques only apply at the
very last stage, on the decompiled/lifted result. However, Patchestry’s approach
to decompilation is multi-level: decompilation progresses through a stage of
increasingly high-level IRs. By taking a multi-level approach, Patchestry can
instrument contracts at various stages of the process.

5. Verification of contracts. To ensure the reliability of patched code along
with associated contracts, Patchestry offers a toolset for generating output
compatible with both static and dynamic analysis tools. The optimal choice for
this purpose is LLVM IR, given its verification confirms the fulfillment of
contracts before its compilation. Patchestry allows for easy integration of
LLVM-based analysis tools such as KLEE or SeaHorn, automating the verification
process.

6. Patchestry formulates the patch by compiling developer-edited decompiled
function(s), and packages the patch for use by a binary patching tool.
Patchestry will utilize a pre-existing tool, such as Patcherex or OFRAK, to
enact the patch process, creating a new version of the binary.

7. Finally, the developer will load the new version of the binary onto the
device. How the developer loads the new version of the binary is not part of the
project.

## Architecture

The Patchestry design places a strong emphasis on modularity and seamless
developer interaction. The developer plays a key role, providing the binary
pieces to be patched, a patch description, and instructions on how to apply
these patches using the meta-programming framework (meta-patches). Contracts are
similarly specified and applied by instrumentation using the same meta-language.
Utilizing state-of-the-art tools, we perform decompilation and program analysis.

A significant architectural innovation is the MLIR Tower of IRs, which serves as
the connecting element. This tower facilitates the association of
representations between decompiled programs, such as from P-Code and compilable
and structured representations like LLVM IR. The tower's modularity allows for
the specification of any DSL for the decompiled program, with the only
requirement being the translation of this DSL to a layer of the tower. In our
case, Ghidra's P-Code serves as a suitable starting point layer. However, this
modular design allows new decompilers to be integrated into Patchestry in the
future while preserving the rest of the architecture.

Utilizing the same representation (MLIR dialects) for both the decompiled binary
and the compiled patched version facilitates seamless instrumentation and
inlining of patches, ultimately producing a patched MLIR (Tower of IRs). The
tower's various abstraction layers enable precise specification of points of
interest, surpassing the limitations of a single representation. Additionally,
the tower abstracts away from the decompiled representation (P-Code),
facilitating modular design in the future.

Contract handling follows a similar pattern. Described in a C-like language,
contracts can take the form of static or runtime assertions or error handlers.
These are inserted into the code while it is in the IR Tower form. Runtime
checks are then compiled and remain in the patched binary. Static contracts are
checked using a formal verifier. The flexibility to invent new contract
mechanisms according to specific needs is a key feature.

In the verification phase, which is the final step, Patchestry is designed to
accommodate various verification methods. The Tower allows to produce a
customized representation for the analysis, but it is advisable to stick to the
same representation as the compilation (such as LLVM IR) to prevent errors
during translation. Slicing the codebase into independent parts influenced by
the patch makes LLVM-based static analysis of the representation with contracts
tractable. We expect that most of the patches being local influence only a small
part of the program, therefore using the dependency analysis, we can isolate the
part of the program that needs to be verified.