% Обновление в июле 2011
% добавлен абзацный отступ в аннотацию статьи
% убрано KeyWords (закомментировано)
% убрано PACS (закомментировано)
% utf-8 в заголовочном комментарии данного файла
% аннотация теперь опциональна
% убрана принудительная капитализация заголовка
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  LaTeX class for preparing papers to "Avtomatika i Telemekhanika"
%%  (under MikTeX 1.2c--2.4 platform)
%%  charset=utf-8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}[1998/12/01]
\ProvidesClass{a&t}[2005/05/07 alpha v2.1
LaTeX class for preparing papers to "Avtomatika i Telemekhanika"]
\ClassWarningNoLine{a&t}{This class requires MikTeX 1.2c--2.4}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}


\ProcessOptions
\LoadClass[a4paper]{article} % стандартный класс
\RequirePackage[utf8]{inputenc} % в кодировке UTF-8
\RequirePackage[T2A,OT1]{fontenc} % шрифты
\RequirePackage[russian]{babel} % русификация
\RequirePackage[leqno]{amsmath} % расширенные возможности для задания формул
\RequirePackage{amsfonts} % то же самое
\RequirePackage{amssymb} % то же самое
\RequirePackage{ifthen} % расширенные возможности логических операций
\RequirePackage{calc} % расширенные возможности арифметических операций
\RequirePackage{myruss} % !!! подправленный стиль Львовского (ставит точки номеров разделов)
\RequirePackage{ruslist} % !!! перечни
\RequirePackage{verbatim} % для возможности закомментировать кусок текста \begin{comment} ... \end{comment}
\RequirePackage{160-240} % !!! размеры полей

%%  !!! --- так помечены нестандартные пакеты

%%  Сколь угодно жидкие строки
\sloppy

\setlength\parskip{2\p@ \@plus \p@}

%%  Знаки нестрогих равенств (как принято у нас)
\def\le{\leqslant}
\def\leq{\leqslant}
\def\ge{\geqslant}
\def\geq{\geqslant}

%%  Аннотация
\renewenvironment{abstract}%
{\vspace{6pt}
\begin{center}\begin{tabular}{p{0.8\textwidth}}\small\indent}%
{\end{tabular}\end{center}\vspace{12pt}}

%%  Список литературы
\def\@biblabel#1{#1.\hfill}
\addto\extrasrussian{%
\def\bibname{\rm\center СПИСОК ЛИТЕРАТУРЫ\small}
\def\refname{\rm\center СПИСОК ЛИТЕРАТУРЫ\small}}

%%  Приложение
\newcounter{isappendix}
\setcounter{isappendix}{0}
\def\appendix#1%
{\section*{\hfill\sl П\,Р\,И\,Л\,О\,Ж\,Е\,Н\,И\,Е  \ #1}%
\refstepcounter{isappendix}}
\@addtoreset{equation}{isappendix}
\def\theequation%
{{\ifnum \arabic{isappendix} > 0 {\mbox{П.}\arabic{equation}}%
\else{\arabic{equation}}\fi}}

%%  Макросы для определения заголовка
\def\year#1{\gdef\@year{#1}} %% год
\def\authors#1{\gdef\@authors{#1}} %% авторы, место работы, занимаемая должность, адрес электронной почты
\def\thanks#1{\gdef\@thanks{#1}} %% благодарности (ссылка на грант и т.п.)
\def\title#1{\gdef\@title{#1}} %% название статьи
%\def\abstract#1{\gdef\@abstract{\parbox{140mm}{\small$\quad\;\,$#1}}} %% текст аннотации

%%  Заголовок
\renewcommand{\maketitle}{%
%{\noindent \leftline { PACS~\@PACS
%                             }}\vskip4mm
\begin{center}
\normalsize\bfseries\copyright~\@year~г.\qquad%
\@authors\\[3ex]
\large\bfseries{\@title}\footnote{\@thanks}\\[3ex] %
\normalsize\normalfont
% \vspace*{1ex}
% \@abstract
\end{center}
%\vspace*{1ex}
%\hspace{\parindent}{\bf Ключевые слова:}~\@KeyWords
 }
\def\PACS#1{\gdef\@PACS{#1}}
\def\KeyWords#1{\gdef\@KeyWords{#1}}
\newcommand{\AdditionalInformation}[3]{%
{\hangafter=1%
\parindent=0pt%
\hangindent=5mm%
{#1},~{\it #2\/},~{\tt#3}
\par
}}

%%  Точки после номеров разделов
\def\@aftersepkern{\kern-.5em}
\def\postsection{.\@aftersepkern}
\def\postsubsection{.\@aftersepkern}
\def\postsubsubsection{.\@aftersepkern}
\def\postparagraph{.\@aftersepkern}
\def\postsubparagraph{.\@aftersepkern}
\def\presection{}
\def\presubsection{}
\def\presubsubsection{}
\def\preparagraph{}
\def\presubparagraph{}
\def\sectioncenterstyle{\centering}
\def\subsectioncenterstyle{\centering}

%%  Точки после номеров теорем
\def\afterthmseparator{.}
\def\@begintheorem#1#2{%
\setlength{\topsep}{1pt}%
\setlength{\partopsep}{0mm}%
\trivlist\item[\hskip\labelsep{\rm
#1\ #2\unskip\afterthmseparator}]\itshape}
\def\@opargbegintheorem#1#2#3{%
\setlength{\topsep}{1pt}%
\setlength{\partopsep}{0mm}%
\trivlist\item[\hskip\labelsep{\rm
#1\ #2\ {\rm{#3}}\afterthmseparator}]\itshape}

%%  Раздел
\renewcommand\section{\@startsection{section}{1}{\z@}%
{18\p@ \@plus 4\p@ \@minus 4\p@}%
{8\p@ \@plus 2\p@ \@minus 1\p@}%
{\normalfont\normalsize\center\bfseries\boldmath}}

%%  Подраздел
\renewcommand\subsection{\@startsection{subsection}{2}{\parindent}%
{18\p@ \@plus 4\p@ \@minus 4\p@}%
{4\p@ \@plus 2\p@ \@minus 1\p@}%
{\normalfont\normalsize\center\itshape}}

%%  Теорема
\newtheorem{theorem}{\hspace{\parindent}\sl{Т\,е\,о\,р\,е\,м\,а\,}}

%%  Следствие
\newtheorem{corollary}{\hspace{\parindent}\sl{С\,л\,е\,д\,с\,т\,в\,и\,е\,}}

%%  Лемма
\newtheorem{lemma}{\hspace{\parindent}\sl{Л\,е\,м\,м\,а\,}}

%%  Утверждение
\newtheorem{statement}{\hspace{\parindent}\sl{У\,т\,в\,е\,р\,ж\,д\,е\,н\,и\,е\,}}

%%  Предложение
\newtheorem{proposition}{\hspace{\parindent}\sl{П\,р\,е\,д\,л\,о\,ж\,е\,н\,и\,е\,}}

%%  Алгоритм
%\newtheorem{algorithm}{\hspace{\parindent}\sl{А\,л\,г\,о\,р\,и\,т\,м\,}}

%%  Определение
\newtheorem{definition}{\hspace{\parindent}\sl{О\,п\,р\,е\,д\,е\,л\,е\,н\,и\,е\,}}

%%  Задача
\newtheorem{problem}{\hspace{\parindent}\sl{З\,а\,д\,а\,ч\,а\,}}

%%  Доказательство
\newenvironment{proof}
{\vspace{1pt}\par{\sl%
Д\,о\,к\,а\,з\,а\,т\,е\,л\,ь\,с\,т\,в\,о.\,\ }}%
{\noindent\vspace{1pt}}

%%  Доказательство теоремы
\newenvironment{proofoftheorem}[1]
{\vspace{1pt}\par{\sl%
Д\,о\,к\,а\,з\,а\,т\,е\,л\,ь\,с\,т\,в\,о\, т\,е\,о\,р\,е\,м\,ы\, #1.}}%
{\noindent\vspace{1pt}}

%%  Доказательство следствия
\newenvironment{proofofcorollary}[1]
{\vspace{1pt}\par{\sl%
Д\,о\,к\,а\,з\,а\,т\,е\,л\,ь\,с\,т\,в\,о\, с\,л\,е\,д\,с\,т\,в\,и\,я\, #1.}}%
{\noindent\vspace{1pt}}

%%  Доказательство леммы
\newenvironment{proofoflemma}[1]
{\vspace{1pt}\par{\sl%
Д\,о\,к\,а\,з\,а\,т\,е\,л\,ь\,с\,т\,в\,о\, л\,е\,м\,м\,ы\, #1.}}%
{\noindent\vspace{1pt}}

%%  Доказательство утверждения
\newenvironment{proofofstatement}[1]
{\vspace{1pt}\par{\sl%
Д\,о\,к\,а\,з\,а\,т\,е\,л\,ь\,с\,т\,в\,о\, у\,т\,в\,е\,р\,ж\,д\,е\,н\,и\,я\, #1.}}%
{\noindent\vspace{1pt}}

%%  Доказательство предложения
\newenvironment{proofofproposition}[1]
{\vspace{1pt}\par{\sl%
Д\,о\,к\,а\,з\,а\,т\,е\,л\,ь\,с\,т\,в\,о\, п\,р\,е\,д\,л\,о\,ж\,е\,н\,и\,я\, #1.}}%
{\noindent\vspace{1pt}}

%%  Алгоритм
\newtheorem{alg}{\hspace{\parindent}\sl{А\,л\,г\,о\,р\,и\,т\,м\,}}%
\newenvironment{algorithm}[1][\unskip]{\begin{alg}[#1]\em}{\end{alg}}%
\def\thealgorithm{\thealg}

%%  Замечание
\newtheorem{rem}{\hspace{\parindent}\sl{З\,а\,м\,е\,ч\,а\,н\,и\,е}}%
\newenvironment{remark}[1][\unskip]{\begin{rem}[#1]\em}{\end{rem}}%
\def\theremark{\therem}

%%  Пример
\newtheorem{exmpl}{\hspace{\parindent}\sl{П\,р\,и\,м\,е\,р\,}}%
\newenvironment{example}[1][\unskip]{\begin{exmpl}[#1]\em}{\end{exmpl}}%
\def\theexample{\theexmpl}


%%  Ссылка на элемент перечня
\def\itemref#1{{\rm\ref{#1})}}

%%  В окружении multline сдвиг=0
\multlinegap=0pt

\endinput
