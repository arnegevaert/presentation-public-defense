# Public defense

## Preamble: (0m45s)
Voor we beginnen een kleine dienstmededeling: de presentatie gaat volledig in het Nederlands zijn,
maar de slides zijn in het Engels, zodat de Engelstaligen onder ons ook kunnen volgen.
Ik ga dit nu nog even meedelen aan de Engelstaligen, en dan kunnen we beginnen.
Hello everyone, I have some sad news: I decided to do this presentation entirely in Dutch.
I know, this is very disappointing, but please understand my situation:
today I have the opportunity to talk to my friends and family about my work for *40 minutes*,
and they *can't run away.*
This is a once in a lifetime opportunity, so please forgive me for taking it.
That being said, the slides are in English and I have added subtitles wherever possible,
so you should be able to follow using the slides.

## Introductie: (8m15s)

### Intro part 1 (1m30s)
Om te beginnen, moeten we even terug naar iedereen zijn/haar favoriete plek en meest dierbare herinnering:
de wiskundeles in het derde middelbaar.
Ik voel de spanning al een beetje toenemen in de zaal, maar er is geen reden om bang te zijn.
Ik heb namelijk gemerkt dat de meeste mensen die een trauma hebben aan wiskunde
eigenlijk vooral een trauma hebben aan de wiskunde*toets.*
Vandaag is er geen toets, dus ook geen reden om bang te zijn.

We beginnen met het concept *functie.*
Een echte informaticus denkt in termen van in- en uitvoer,
en we kunnen een functie ook zo bekijken:
een soort "machine" die een invoer neemt en een uitvoer teruggeeft.

Onze "machine" heeft ook een soort "programmacode" of "instructies"
die duidelijk maken wat de machine met zijn invoer moet doen.
We schrijven die programmacode in een vergelijking:
aan de linkerkant hebben we f van x, dus f is onze machine en x is de invoer,
en aan de rechterkant schrijven we wat we willen dat f doet met x.
In dit geval gaat f het getal x vermenigvuldigen met 2 en er dan 1 van aftrekken.

We kunnen dit eens testen: als we het getal 3 meegeven met f,
dan krijgen we inderdaad 2 maal 3 min 1, en dat is 5.
We kunnen hetzelfde doen met -1, en dan krijgen we 2 maal -1 min 1, en dat is -3.

We kunnen daar een grafiekje van maken.
Hier zien we onze twee punten terug, die komen elk overeen met een invoer op de X-as
en de bijhorende uitvoer (de hoogte).
Als we dit herhalen voor alle getallen, krijgen we deze lijn te zien.

### Intro part 2 (2m30s)
Laten we nog eens kijken naar onze formule.
Er zijn twee getallen die onze formule eigenlijk vastleggen: die 2 en die -1.
Als we een beetje prutsen aan die 2, dan zien we dat de richting van onze lijn verandert.
Als we prutsen aan die -1, dan gaat onze lijn naar boven en beneden.
Die 2 en die -1, we noemen die ook *parameters.*

Machine learning is eigenlijk niets anders dan de computer 
automatisch laten zoeken naar een goeie waarde van die parameters
zodat onze functie voor iets nuttigs gebruikt kan worden.
Wat bedoel ik daarmee?
Stel dat we bijvoorbeeld op basis van de temperatuur willen kunnen voorspellen 
hoeveel energie we verbruiken in ons huis.
Als het buiten koud is, gaan we waarschijnlijk meer energie nodig hebben om te verwarmen.

Wat we kunnen doen, is op een aantal dagen meten hoe warm het buiten is,
dat is onze X-as,
en dan meten hoeveel energie we hebben verbruikt.
Hier heb ik dat gedaan voor 6 dagen, dus we hebben 6 punten op onze grafiek.
Stel nu dat het weerbericht zegt dat het morgen 20 graden gaat worden.
Ik wil graag voorspellen hoeveel energie ik morgen ga verbruiken.
We kunnen dat doen door een lijn te trekken die zo mooi mogelijk door deze data loopt,
en dan te kijken hoe hoog de lijn is op temperatuur 20.

We starten met een willekeurige waarde voor de 2 parameters.
Dat geeft ons deze lijn, die duidelijk geen goede beschrijving is van de data.
We kunnen ook uitdrukken hoe slecht die lijn is met een *foutfunctie.*
Hoe die foutfunctie werkt, doet er nu even niet toe.
Alles wat we moeten weten is, hoe slechter de lijn, hoe groter de foutfunctie.

Nu hebben we alles wat we nodig hebben om machine learning te doen.
We moeten gewoon zeggen aan de computer,
ga op zoek naar parameterwaarden die ervoor zorgen dat die foutfunctie zo klein mogelijk is.
Dat noemen we ook *trainen.*
Laten we dat eens doen, en voila, de lijn gaat mooi door de data.
We zien ook dat de foutfunctie nu veel kleiner is dan daarnet.
Zo'n functie waarvan de computer zelf de parameters heeft gekozen,
noemen we in machine learning ook een *model.*

Nu denk je misschien, maar allez, we hadden die lijn nu toch ook wel met het blote oog kunnen tekenen.
En dat klopt, maar dat gaat niet altijd lukken.
Stel bijvoorbeeld dat we niet alleen de temperatuur bijhouden,
maar ook hoeveel mensen er thuis zijn.
Dan hebben we twee getallen als invoer, oftewel twee *invoerveranderlijken.*
Elk van deze veranderlijken krijgt een as, en dan krijgen we deze 3D-grafiek.
Ieder punt komt overeen met een waarde voor temperatuur en aantal mensen.
De hoogte van ieder punt is dan het energieverbruik.
Hier kunnen we nog altijd een model door tekenen: we krijgen dan een vlak
in plaats van een lijn.

Stel nu dat we ook nog meten hoe hard de zon schijnt, want er liggen zonnepanelen op ons dak.
Nu moeten hebben we dus 3 invoerveranderlijken.
Dit kunnen we niet meer visueel weergeven, aangezien we jammer genoeg leven in 3 dimensies,
en in totaal 4 dimensies zouden nodig hebben: 3 voor de invoer en 1 voor de uitvoer.
Maar voor de computer is dit geen probleem.


### Intro part 3 (2m45s)
Het leuke aan dit model is dat we gewoon kunnen kijken naar de "programmacode",
namelijk de vergelijking,
en dus kunnen zien wat dit model "doet."
We zien bijvoorbeeld dat de parameters voor temperatuur en zon negatief zijn:
hoe warmer en hoe meer zon, hoe minder energieverbruik.
De parameter voor aantal personen is positief:
hoe meer mensen thuis, hoe meer energieverbruik.

Er is wel nog een probleempje met ons model.
Om dat te zien, gaan we eventjes terug naar een model met maar 1 invoergetal,
zodat we ons grafiekje kunnen zien.
Ons model kan enkel rechte lijnen tekenen.
Dat gaat niet altijd voldoende zijn, en je kan het hier eigenlijk ook al zien:
als het buiten voldoende warm wordt,
dan gaan we zogezegd negatieve energie verbruiken.
Met zonnepanelen kan dat misschien nog in principe,
maar als we bijvoorbeeld airco in huis hebben en buiten is het 40 graden,
dan denk ik dat ons energieverbruik opnieuw omhoog gaat gaan.

Als onze punten geen mooie rechte lijn vormen,
dan kunnen we ons model flexibeler maken door parameters toe te voegen.
Stel bijvoorbeeld dat onze data er zo uitziet. *(kwadratisch)*
We kunnen hier niet echt een rechte lijn door tekenen,
maar als we een parameter toevoegen aan ons model,
dan lukt het wel: nu kan ons model rechte lijnen en dit soort kromme lijnen tekenen.
Maar: nu moeten we wel 3 parameters laten kiezen door de computer,
en onze formule wordt ook een beetje moeilijker om te interpreteren.

Stel nu dat onze data er zo uitziet. *(complexe data, NN example)*
We kunnen opnieuw een model maken en de parameters laten kiezen door de computer,
en dat ziet er dan zo uit.
Laten we nu eens kijken naar de programmacode van ons model.
Oei, er is precies iets verkeerd met mijn slides...
Ah nee, ok.
Wat voor model is dit? 
Dit is een *neuraal netwerk,* een speciaal soort model dat we veel gebruiken in machine learning
en waarvan we kunnen kiezen hoeveel parameters het heeft.
Hoe meer parameters, hoe complexer de vormen die het model kan tekenen.
Dit neuraal netwerk heeft 100 parameters, en we zien dus dat het redelijk moeilijk wordt
om de formule echt te verstaan.

Hoeveel parameters zitten er dan in een "echt" machine learning-model, zoals ChatGPT?
Toen ChatGPT 2 jaar geleden uitkwam, gebruikte dat het GPT3.5-model.
Dat model heeft *175* parameters.
Sorry, 175 *miljard* parameters.
Hoeveel parameters er in de nieuwste versie zitten, willen de makers ons niet vertellen,
maar het wordt geschat rond de 1.75 *biljoen*, dus nog eens ongeveer 10 keer zoveel.

### Intro part 4 (1m20s)
Als we alle 1.75 biljoen parameters van ChatGPT op standaard A4 printerpapier zouden zetten, *(2+1+4+1 characters/param, 2000 characters/page, 0.1mm per page => 700km)*
dan zou de stapel papier ongeveer 700km hoog zijn,
oftewel ongeveer de afstand van deze zaal tot in Berlijn.
Als je 2024 jaar geleden, naast de kribbe van het kindeke Jezus,
het model was beginnen opschrijven aan 1 parameter per seconde zonder pauzeren,
dan had je vandaag ongeveer 3.6% van het werk gedaan.
Zo'n groot model kunnen we natuurlijk niet meer verstaan door naar de formule te kijken.

Wat we wel nog kunnen doen, is het model beschouwen als een soort zwarte doos,
waar we invoer aan kunnen geven en uitvoer van terugkrijgen.
Het is nog altijd gewoon een functie.
We kunnen dus nog altijd een beetje prutsen aan de invoerwaarden,
en zien hoe de uitvoer verandert.
Als we prutsen aan een bepaalde invoer en de uitvoer verandert sterk,
dan kunnen we zeggen dat die invoer een sterke invloed heeft.
Als we eraan prutsen en de uitvoer blijft gelijk,
dan is die invoer misschien niet zo belangrijk.

Zo'n manier van inzicht kweken in een model
noemen we een *attributie-gebaseerde verklaring*,
of gewoon *attributies.*
Wiskundig gezien komt het erop neer dat we, op een of andere manier,
een score berekenen voor iedere invoerveranderlijke
die ons zegt hoe *belangrijk* die veranderlijke is.
In ons voorbeeld van daarnet konden we de parameters zelf beschouwen als attributies,
want die zeggen ons wat de invloed is van iedere veranderlijke op de uitvoer.

## Benchmark (6m20s)

### Benchmark part 1 (3m20s)
Dit brengt ons nu naar het eerste grote experiment dat ik gedaan heb voor mijn doctoraat.
We focussen ons eventjes op het classificeren van afbeeldingen.
Dat is een typisch probleem dat we kunnen oplossen met machine learning:
we hebben een hoop foto's, bijvoorbeeld hier foto's van kleine handgeschreven cijfers,
dat zijn onze invoerwaarden.
Voor iedere foto hebben we ook een "label" dat zegt welk cijfer er op de foto staat,
dat is onze uitvoerwaarde.
Als we dus een model maken dat die uitvoerwaarde kan voorspellen,
dan hebben we in feite een programma dat automatisch kan "zien"
welk cijfer er in een foto staat.

Hoe kunnen we nu zo'n foto als invoer geven aan een model?
Simpel: iedere pixel in de foto is een getal dat weergeeft hoe helder die pixel is.
In een kleurenfoto is iedere pixel 3 getallen: de hoeveelheid rood, groen en blauw in de pixel.
Dus als we zoals hier een klein zwart-wit fotootje hebben van 28 op 28 pixels,
dan heeft ons model 784 invoerveranderlijken nodig.

Attributies zijn heel handig in dit geval.
Waarom? Omdat iedere invoerveranderlijke, dus iedere pixel, een score krijgt.
We kunnen dus iedere pixel een kleurtje geven die die score weergeeft,
en dan krijgen we dit soort visualizatie.
Dit zegt ons, voor een bepaalde foto, welke pixels het *belangrijkst* waren.
Hier zien we bijvoorbeeld dat de pixels van de nul zelf belangrijk waren,
maar ook de pixels in het midden:
als die een andere waarde hadden, dan was onze nul misschien een acht.
Merk op: nu hebben we een score voor 1 specifieke foto.
We noemen dat ook *lokale* attributies.
Daarnet, in ons model om energie te voorspellen,
beschreven de scores de invloed van de veranderlijken in het algemeen,
los van een specifieke meting.
Dat noemen we dan *globale* attributies.

Wat is nu het nut van zulke lokale attributies?
Wel, we kunnen ze bijvoorbeeld gebruiken om te kijken of het model wel geleerd heeft
wat we willen dat het leert.
Een kleine anecdote.
Er was eens een groep onderzoekers die een model wou maken dat op basis van een foto van een plek op de huid
kon voorspellen of die plek kwaadaardig of goedaardig zou zijn.
Dat zou het veel makkelijker maken om bijvoorbeeld huidkanker vroeg op te sporen.
Ze hadden veel data verzameld van zowel goedaardige als kwaadaardige plekken op de huid,
en als ze hun model testten op die data, werkte alles perfect:
het model kon redelijk goed voorspellen of een plek kwaadaardig of goedaardig was.
Maar, als ze het model testten op nieuwe data, waren de voorspellingen plots helemaal nutteloos:
we konden eigenlijk bijna even goede voorspellingen maken door gewoon te gokken.

De onderzoekers gebruikten attributies om te zien waar hun model naar keek,
en wat bleek: als er markeringen waren gemaakt op de huid, dan keek het model vooral naar die markeringen.
Blijkbaar, als een dokter vindt dat een plek kwaadaardig is,
dan gaat die markeringen tekenen op de huid die tonen aan de chirurg hoe de plek verwijderd moet worden.
Het model ging dus gewoon op zoek naar van die markeringen.
Als er geen markeringen op de huid staan, dan voorspelde het model dat de plek niet kwaadaardig was.
Natuurlijk is dat redelijk zinloos,
want als er al markeringen op de huid staan,
dan is de diagnose al gebeurd en is de voorspelling van het model dus niet meer nuttig.

### Benchmark part 2 (3m)
Er zijn heel veel mensen die onderzoek doen naar goeie manieren om zo'n attributies te berekenen,
en dus zijn er heel veel manieren ontwikkeld om dat te doen.
Al die manieren geven ons verschillende attributies.
Dat geeft ons een nieuw probleem: welke methode moeten we nu gebruiken?
Als we op een of andere manier de kwaliteit van de attributies zouden kunnen meten,
dan kunnen we de beste methode eruit kiezen.

Er zijn heel veel mensen die onderzoek doen naar goeie manieren om de kwaliteit van attributies te meten,
en dus zijn er heel veel manieren ontwikkeld om dat te doen.
Al die kwaliteitsmetrieken zijn ontworpen om te meten hoe *correct* de attributies zijn:
zijn de "belangrijkste" pixels volgens de attributies ook echt belangrijk?
Dat gaf ons een simpel idee:
we verzamelen een paar datasets,
een paar methoden om attributies te genereren,
en een paar kwaliteitsmetrieken.
Voor elk van de datasets maken we een model,
genereren we attributies op een stuk of 100 foto's,
en meten we de kwaliteit van die attributies.

Hier zien we een deel van de resultaten van dat experiment.
Elke grafiek komt overeen met een dataset.
De kolommen zijn de methodes, de rijen de kwaliteitsmetrieken.
De grootte en donkerheid van de groene vierkantjes toont,
voor iedere methode en iedere metriek,
hoe goed de kwaliteit is van die methode volgens die kwaliteitsmetriek.

Wat zien we?
De kwaliteit van attributies hangt af van de dataset waarop het model getraind is.
Dat zegt ons dus dat we niet 1 methode kunnen aanduiden die de "beste" is en gewoon altijd die gebruiken.
We komen er helaas niet zo gemakkelijk vanaf.
Maar wat zien we nog: de kwaliteit van de attributies hangt ook af van de kwaliteitsmetriek!
De ene metriek kan ons zeggen, attributiemethode A is beter dan attributiemethode B,
terwijl de andere metriek juist het omgekeerde zegt.
Dat is vreemd!
Alle metrieken waren gemaakt om te meten *hoe correct* de attributies zijn.
Hoe kan het dan dat ze elkaar tegenspreken?

De conclusie is:
de kwaliteitsmetrieken meten toch verschillende eigenschappen van de attributies,
en niet gewoon "de correctheid."
We kunnen de "correctheid" niet uitdrukken in 1 simpel getal.
De vraag is nu, wat zijn die onderliggende eigenschappen precies?
Hoe kunnen we de verschillende attributiemethodes met elkaar vergelijken?

## Removal-based attribution methods (12m15s)

### RBAM part 1 (2m00s)
Laten we even een stapje terug nemen, en nadenken hoe we attributies kunnen berekenen.
Stel dat we een model hebben dat voorspelt of iemand een hoog of laag risico heeft op diabetes,
op basis van lengte, gewicht, leeftijd, bloeddruk en cholesterol.
We hebben dit model getraind door deze 5 eigenschappen van een hoop mensen te meten,
dat zijn de invoerveranderlijken,
en dan na te gaan of die mensen diabetes hebben,
dat is de uitvoer.
Dan hebben we, net zoals in het begin van de presentatie,
aan de computer gevraagd om de parameters van het model zo in te vullen
dat de uitvoer overeenkomt met de kans op diabetes,
gegeven de 5 metingen van de patient.
Onze attributies gaan dus moeten aanduiden hoe belangrijk elk van deze 5 invoervariabelen nu is.

De eerste keuze die we moeten maken is:
waarvoor precies willen we de attributies berekenen?
Willen we dat de attributies ons tonen hoe belangrijk iedere invoervariabele is
voor een specifieke persoon,
zoals bij de foto's van daarnet,
of willen we weten hoe belangrijk iedere variabele is voor het model in het algemeen,
zoals bij ons voorbeeld van het energieverbruik?
We kunnen bijvoorbeeld ook kiezen om te meten hoe belangrijk iedere variabele is voor
de *correctheid* van ons model,
herinner u de foutfunctie die zegt hoe "goed" het model is in het voorspellen van de data.
Een veranderlijke kan bijvoorbeeld een positieve invloed hebben op de uitvoer van het model,
met andere woorden,
als we de waarde van de veranderlijke verhogen, gaat de uitvoer ook omhoog,
maar dezelfde veranderlijke kan tegelijk een negatieve invloed hebben op hoe correct het model is,
met andere woorden,
als we de waarde verhogen, wordt de voorspelling van het model minder juist.
Deze keuze is het eerste ingredient van onze attributiemethode: het **doelwit,**
oftewel, waarvoor precies willen we een verklaring?

### RBAM part 2 (3m00s)
Stel nu dat we een patient hebben waarvoor het model zegt dat er een hoog risico is op diabetes,
en we willen dat onze attributies ons tonen hoe belangrijk iedere variabele is voor die voorspelling.
Met andere woorden, ons doelwit is gekozen: het is de uitvoer van het model voor deze specifieke patient.
Wat kunnen we dan doen om bijvoorbeeld te meten hoe belangrijk de veranderlijke "leeftijd" is voor die voorspelling?
Het liefst zouden we eigenlijk vragen aan het model: *als we leeftijd niet zouden kennen, wat zou dan je voorspelling zijn?*
Als we dat zouden kunnen doen, dan kunnen we kijken naar hoe hard de voorspelling verandert,
en als die veel verandert, dan is de leeftijd blijkbaar belangrijk.

Jammer genoeg werkt dat niet, want we kunnen niet zomaar een variabele niet invullen in het model.
Ons model kan ons dan geen uitvoer geven.
We moeten dit probleem dus proberen te omzeilen.
Een eerste idee zou bijvoorbeeld kunnen zijn om de waarde 0 mee te geven als leeftijd.
Maar dat lost het probleem niet echt op.
Wij zouden dat misschien kunnen interpreteren als "geen leeftijd",
maar ons model weet dat niet.
Voor ons model is dit gewoon een pasgeboren baby... van 85kg en 195cm.
Waarschijnlijk heeft het model niet veel zo'n pasgeboren baby's gezien in zijn data.

Een andere manier kan bijvoorbeeld zijn om niet 1 waarde in te vullen voor leeftijd,
maar alle mogelijke waarden die we gezien hebben in de data.
Als we bijvoorbeeld 1000 patienten hadden in de data, dan berekenen we de uitvoer van het model 1000 keer,
telkens voor een andere leeftijd.
Dan kunnen we kijken naar de gemiddelde uitvoer van het model op die 1000 voorbeelden,
en zien of dat veel verschilt van de uitvoer op de oorspronkelijke patient.
Dit is al beter en wordt in het echt ook veel gedaan,
maar het is nog altijd niet perfect:
we kunnen namelijk nog altijd rare combinaties krijgen.
Als er in onze dataset bijvoorbeeld ook kinderen zaten, dan gaan we hier patienten verzonnen hebben
die bijvoorbeeld 9 jaar oud zijn, maar toch 195cm lang.
Het probleem daarmee is dat ons model waarschijnlijk nooit zo'n lange kinderen gezien heeft in de data.
Als we dan toch zo'n extreem lang kind meegeven aan het model,
dan hebben veel modellen de neiging om extreme, willekeurige uitvoerwaarden terug te geven.
Dat kan dus een invloed hebben op ons gemiddelde.

Er bestaan nog veel meer gesofisticeerde manieren om dit probleem op te lossen,
en elk van die manieren heeft zijn voor- en nadelen.
Maar over het algemeen proberen we telkens hetzelfde te doen:
we proberen een variabele uit de invoer te **verwijderen.**
Dit is het tweede ingredient van onze attributiemethode.

### RBAM part 3 (2m15s)
Stel nu dat we een methode hebben gekozen om variabelen te verwijderen,
bijvoorbeeld door het gemiddelde te nemen over de waarden in de dataset.
Het probleem van de buitenaards lange kinderen laten we eventjes voor wat het is.
Nu kunnen we attributies berekenen:
we verwijderen elk om beurt iedere invoervariabele,
en kijken naar het verschil in uitvoer.
Dit verschil in uitvoer is dan onze attributiescore voor die variabele. Top!

Maar ik vrees dat we nog altijd niet helemaal klaar zijn.
Stel dat ons model eigenlijk heel simpel is: het voorspelt een verhoogde kans als
de cholesterol hoger is dan 100 of de bloeddruk hoger is dan 130.
Inderdaad, onze patient heeft een cholesterol van 110 en een bloeddruk van 135.
Maar wat gaat er gebeuren als we onze attributies berekenen?
Als we cholesterol "verwijderen", dan gaan we allemaal verschillende waarden invullen,
maar dit gaat de uitvoer nooit veranderen,
want de bloeddruk is nog altijd 135.
Als we bloeddruk "verwijderen", gaat de uitvoer ook nooit veranderen,
want de cholesterol is nog altijd 110.
Met andere woorden, onze attributies gaan zeggen dat *geen enkele* veranderlijke belangrijk was!

Wat is er hier misgelopen?
Het probleem is dat we naar iedere veranderlijke individueel kijken,
terwijl het model naar cholesterol en bloeddruk *tegelijk* kijkt.
Wat we dus eigenlijk zouden moeten doen,
is niet alleen iedere veranderlijke apart verwijderen,
maar ook alle mogelijke groepen van veranderlijken verwijderen.
Dat kunnen we natuurlijk doen, maar dan hebben we een nieuw probleem:
voor een verzameling van 5 veranderlijken zijn er 32 mogelijke groepen (of deelverzamelingen).
We hebben nu dus 32 uitvoerwaarden voor die 5 veranderlijken.
We moeten dus nog op een of andere manier die 32 uitvoerwaarden "samenvatten" in 5 scores,
namelijk 1 voor iedere veranderlijke.
Dit is het derde en laatste ingredient van onze attributiemethode:
de manier van **aggregeren,**
of met andere woorden:
hoe vatten we de 32 uitvoerwaarden samen in 5 scores.

### RBAM part 4 (1m30s)
We hebben dus 3 ingredienten voor een attributiemethode:
het doelwit, waar we precies een verklaring voor willen maken,
een manier om veranderlijken te verwijderen uit het model,
en een manier om de waarden samen te vatten in 5 scores.
Wat ik nu heb aangetoond in het tweede deel van mijn onderzoek is dat niet alle,
maar wel een hele grote hoop bestaande attributiemethodes
eigenlijk allemaal kunnen beschreven worden met die 3 keuzes.

Ik heb dat natuurlijk iets formeler gedaan dan ik het tot nu toe heb uitgelegd.
Maak je geen zorgen: deze formule zegt eigenlijk gewoon hetzelfde als wat ik tot nu toe heb verteld,
maar dan in wiskundetaal.
Er komt geen toets.

Laten we eventjes stilstaan bij deze formule.
Wat ik hier beschrijf is opnieuw een functie, en die functie is onze attributiemethode.
We geven 3 veranderlijken mee aan onze functie:
een model f, een veranderlijke i en een invoer x.
Bijvoorbeeld, het model f kan ons model zijn van daarnet dat het risico op diabetes voorspelt
op basis van 5 eigenschappen.
De veranderlijke i daarnet was "leeftijd",
en de invoer x was de lijst van 5 getallen die overeenkomt met onze patient.
De uitvoer van deze functie is dan een score die zegt hoe belangrijk
de leeftijd (i) van de patient (x) was voor de voorspelling van het model (f).

### RBAM part 5 (3m30s)
Laat ons nu eens kijken naar de rechterkant van de vergelijking.
Hier is ons model f terug.
We hebben hier een nieuw symbool: P_T(f).
Die P_T, dat is opnieuw een functie.
T is een deelverzameling van de variabelen.
P_T is dan een speciale functie: deze neemt het model f, en geeft ons een nieuw model P_T(f) terug.
Dit nieuw model is onafhankelijk van de variabelen in T.
Met andere woorden, P_T *verwijdert* de variabelen in T uit f.
Dat was ons tweede ingredient van daarnet.

Rond die P_T(f) staat nog iets: Phi.
Die Phi is, je raadt het misschien al, weer een functie.
Deze functie neemt P_T(f), dus het model waaruit een aantal veranderlijken verwijderd zijn,
en geeft ons opnieuw een functie terug.
Dit komt overeen met het eerste ingredient van daarnet: het *doelwit.*
Als we bijvoorbeeld een verklaring willen voor de correctheid van het model,
dan zouden we voor Phi de foutfunctie kunnen kiezen.

We hebben dus Phi(P_T(f)), en dat is de eigenschap van het model waar we een verklaring voor willen,
nadat de veranderlijken in T zijn verwijderd.
Dan blijven nog deze alpha's over.
Die alpha's zijn... geen functies, maar gewoon getallen.
Voor een gegeven verzameling T, vermenigvuldigen we alpha_T met die constructie Phi(P_T(f)),
en dan nemen we de som over alle verzamelingen van veranderlijken T.
Die alpha's zeggen ons dus hoe we de verschillende waarden voor de verzamelingen T
moeten samenbrengen in 1 score per veranderlijke:
dit is de manier van *aggregeren,*
ons derde ingredient.

We kunnen deze vergelijking dus zo samenvatten:
voor iedere mogelijke groep van veranderlijken,
verwijder die groep uit het model f,
meet daarvan de eigenschap die ons interesseert (bijvoorbeeld de uitvoer, of de correctheid).
Daarna vermenigvuldigen we elk van deze waarden met een bepaald getal en nemen we de som,
en dat is onze attributiescore voor de veranderlijke i!

Als je goed opgelet hebt, kan je misschien onze methode van daarnet herkennen.
We waren geinteresseerd in de uitvoer van het model f zelf:
dus Phi(f) geeft ons gewoon de functie f terug.
We verwijderden variabelen door alle mogelijke waarden in te vullen en een gemiddelde te nemen van de uitvoer:
dit is P_T.
En we keken naar iedere variabele apart,
dus de alpha's waren 1 voor de lege verzameling,
-1 voor de verzameling die overeenkomt met de variabele i,
en 0 voor al de rest.
Zo krijgen we dus: 1 keer de uitvoer van f zonder variabelen te verwijderen,
min 1 keer de uitvoer van f als i verwijderd is,
of met andere woorden,
het verschil in uitvoer als we i verwijderen.

In mijn doctoraat heb ik nu wiskundig kunnen bewijzen
dat tientallen attributiemethodes die in de laatste 10 a 15 jaar zijn uitgevonden,
allemaal door andere mensen en met andere redeneringen erachter,
allemaal gewoon te schrijven zijn als deze formule.
Dat maakt al die methodes plots enorm veel makkelijker om te vergelijken:
als we twee methodes met elkaar vergelijken,
moeten we dus gewoon de 3 kenmerkende eigenschappen van die methodes naast elkaar leggen.

- [ ] Hier explicieter maken: LIME vs een andere methode, allemaal toch hetzelfde.

## Functional decomposition (6m00s)

### FD part 1 (2m10s)
Er is nog een belangrijk concept dat ik nog niet heb uitgelegd, en dat is *interactie.*
Iedereen die kinderen heeft of ooit met kinderen heeft gewerkt, kent dit fenomeen.
Stel, er zijn 2 kinderen, die we liefkozend Kind A en Kind B noemen.
Kind A en Kind B zijn heel goede vrienden.
Stel nu dat er een speelkamer vol legoblokken ligt, en die moeten opgeruimd worden.
We hebben dan 4 mogelijke situaties.
Als geen enkel kind opruimt, dan zijn er achteraf 0 legoblokken opgeruimd. Simpel.
We kunnen dit schrijven als een functie b:
als we de lege verzameling meegeven aan b, is de uitkomst 0 opgeruimde blokken.
Als Kind A alleen opruimt, dan zijn er achteraf bijvoorbeeld 25 legoblokken opgeruimd.
Dus: b(A) = 25.
Dit noemen we ook het *direct effect* van Kind A.
Op dezelfde manier ruimt Kind B bijvoorbeeld 20 blokken op: b(B) = 20.
Maar, wat gaat er gebeuren als we Kind A en Kind B samen in de speelkamer plaatsen?
Volgens mij gaan er bitter weinig blokken opgeruimd zijn.
In dit geval is het aantal opgeruimde blokken 5, dus b(A,B) = 5.

Wat er gebeurt in deze situatie is een *interactie,* meer bepaald een interactie tussen Kind A en Kind B
waardoor het aantal opgeruimde blokken drastisch daalt.
Wiskundig gezien krijgen we dat b(A,B) kleiner is dan b(A) + b(B),
het aantal blokken dat ze samen opruimen is kleiner dan de som van het aantal blokken die ze elk apart opruimen.
Dit verschil, namelijk 5 - (20 + 25) = -40, is het *interactie-effect* van het samenplaatsen van de 2 kinderen.
Door de 2 kinderen te laten interageren, hebben we het aantal opgeruimde blokken doen dalen met 40.

### FD part 2 (2m20s)
Wat heeft dat nu te maken met attributie?
Wel, herinner u het voorbeeld van het diabetesmodel.
Daar zagen we dat het verwijderen van de veranderlijken cholesterol of bloeddruk apart geen effect had op de uitvoer,
maar als we ze samen verwijderen is er wel een effect.
Dit betekent dat er een *interactie* is tussen die twee veranderlijken,
waardoor we ze niet in isolatie kunnen bekijken.

Er is een fundamentele link tussen het verwijderen van veranderlijken en interacties tussen die veranderlijken.
Laten we eens kijken naar een voorbeeld.
Hier is een functie met 3 veranderlijken. *(f = 2x1 + 4x2 - 3x1x2 + x3 + 5)*
Stel dat we kiezen om veranderlijken te verwijderen door ze op 0 te zetten.
Wat is dan de uitvoer van f als we geen enkele variabele kennen, oftewel, als alle variabelen verwijderd zijn?
We zetten alle variabelen op 0, en we krijgen: 5.
Wat is nu de uitvoer van f als we enkel de variabele x1 kennen?
Dit kunnen we berekenen door alle andere variabelen te verwijderen, zodat enkel x1 overschiet.
We krijgen nu: 2x1 + 5.
Het verschil tussen die twee functies, 2x1, is dan het *direct effect* van x1 op onze functie.
Dit is zoals "Kind 1" dat apart blokken opruimt.

We kunnen hetzelfde doen voor x2 en x3, en we krijgen 4x2 voor x2,
en x3 voor x3.
We hebben nu de "directe effecten" van alle drie de variabelen.
Hoe kunnen we nu de interacties tussen de variabelen meten?
Wel, in het voorbeeld van de kinderen was de interactie tussen Kind A en Kind B het verschil
tussen hun effect als groep, en de som van hun aparte, directe effecten.
We kunnen dit hier ook doen:
om de interactie tussen x1 en x2 te meten, kunnen we het effect van x1 en x2 samen definieren
door alle andere variabelen te verwijderen,
en opnieuw die waarde 5 af te trekken die we krijgen als er geen enkele variabele gekend is.
We krijgen: 2x1 + 4x2 - 3x1x2.
Nu kunnen we kijken naar de som van de directe effecten van x1 en x2, dit is 2x1 + 4x2.
Het verschil is dus: -3x1x2.
Met andere woorden, het interactie-effect tussen x1 en x2 is -3x1x2.

### FD part 3 (1m30s)
Wat hebben we nu zojuist gedaan?
We hebben de functie f gesplitst in een som van kleinere functies.
Elk van deze kleinere functies is afhankelijk van een deelverzameling van de variabelen,
en de precieze manier waarop we f gesplitst hebben
is volledig bepaald door de manier waarop we variabelen verwijderen uit f.
We noemen dit een *additieve decompositie* van f:
"decompositie" omdat we f splitsen in deeltjes,
en "additief" omdat de som van die deeltjes gelijk is aan f.

Nu blijkt dat iedere functie die je kan verzinnen op een gelijkaardige manier gesplitst kan worden
in een som van zo'n kleinere deelfuncties.
Ik geef die decompositie hier schematisch weer: dit is het directe effect van x1,
dit is het interactie-effect tussen x1 en x2, enzovoort.
Daarnet heb ik nog uitgelegd dat,
om een attributiemethode te ontwerpen,
we een manier moeten kiezen om variabelen te verwijderen uit een model.
Wel, in mijn doctoraat heb ik aangetoond dat iedere mogelijke manier om variabelen te verwijderen uit een model,
precies overeenkomt met een additieve decompositie van dat model.
Dat wil dus zeggen dat al die tientallen attributiemethoden waarvan ik daarnet zei
dat ze samengevat kunnen worden met 3 keuzes,
eigenlijk achter de schermen een decompositie maken van het model f,
ook al hadden de originele uitvinders van die methoden dat eigenlijk niet zo bedoeld.
Met andere woorden, als we kijken naar de 3 ingredienten van een attributiemethode,
dan kunnen we eigenlijk ingredient 2,
een manier om variabelen te verwijderen,
vervangen door: een manier om een model op te splitsen in additieve deeltjes.

## PDD-SHAP (6m40s)

### PDD-SHAP part 1 (2m40)
Je kan je nu afvragen, wat is daar nu het praktisch nut van?
Jah, praktisch nut, dat is toch gewoon interessant?
Allez, wie in de zaal had er verwacht dat het naar het einde toe plots over additieve functiedecompositie ging gaan?
Ik niet alleszins.
Maar ok, zelfs naast deze mooie wiskundige link,
die voor mij eigenlijk al interessant genoeg is,
is er ook zelfs nog een praktisch nut.

Er is een bepaalde, bestaande attributiemethode die enorm populair is, en die heet SHAP.
De wiskunde achter die methode is heel geavanceerd,
gebaseerd op een theorie die oorspronkelijk ontwikkeld was om de uitkomst van een economische samenwerking
eerlijk te verdelen onder de leden van een groep.
Het is een heel ingewikkelde methode om uit te leggen,
dus ik ga het niet proberen.
Maar, we kunnen SHAP wel bekijken vanuit ons nieuw wiskundig kader
met de 3 definierende keuzes.

SHAP geeft attributiescores voor de uitvoer van de functie in een bepaald punt,
dus eigenlijk zoals daarnet,
toen we een verklaring wilden voor het risico op diabetes voor een bepaalde specifieke patient.
Dat is ons *doelwit.*
SHAP *verwijdert* variabelen door alle mogelijke waarden in te vullen en het gemiddelde te nemen van de uitvoer,
dat hebben we ook al gezien.
De aggregatie van SHAP is ingewikkelder.
Aangezien SHAP werkt door variabelen te verwijderen,
maakt de methode achter de schermen ook een decompositie.

We kunnen nu de decompositie gebruiken om de SHAP-waarde voor een variabele te berekenen,
laat ons bijvoorbeeld x1 kiezen hiervoor.
De SHAP-waarde voor x1 is het direct effect van x1,
plus alle interacties tussen x1 en een andere variabele, gedeeld door 2,
plus alle interacties tussen x1 en 2 andere variabelen, gedeeld door 3,
enzovoort.
We kunnen SHAP dus eigenlijk samenvatten als:
maak een decompositie van f,
door f te splitsen in een som van kleinere functies,
en verdeel ieder interactie-effect tussen een groep variabelen
gelijk onder die variabelen.
Met andere woorden, de helft van dit interactie-effect gaat naar x1,
de andere helft gaat naar x2.
Een derde van dit interactie-effect gaat naar x1,
een derde gaat naar x2,
en een derde gaat naar x3.

### PDD-SHAP part 2 (2m30s)
Ik heb daarnet gezegd dat SHAP variabelen verwijdert door een gemiddelde te nemen van de uitvoer
voor alle mogelijke waarden van de invoer.
Typisch gaan we niet letterlijk alle mogelijke waarden invullen,
maar gaan we bijvoorbeeld 100 verschillende waarden invullen,
en dan veronderstellen dat dat ongeveer representatief is.

Nu, SHAP kijkt ook naar alle mogelijke deelverzamelingen van variabelen,
iets dat we daarnet in ons diabetesvoorbeeld niet deden.
Herinner u dat we voor 5 variabelen 32 mogelijke deelverzamelingen hebben.
Als we een variabele toevoegen, verdubbelt dat aantal:
met 6 variabelen hebben we 64 mogelijke deelverzamelingen,
7 variabelen geeft ons 128 verzamelingen,
enzovoort.
Om SHAP te berekenen, moeten we dus in principe al die deelverzamelingen verwijderen,
en voor elke deelverzameling moeten we 100 verschillende waarden invullen.
Voor 7 variabelen moeten we dus bijvoorbeeld 12800 keer de uitvoer van het model berekenen.
Dat is heel veel,
en typisch gaan we ook niet alle deelverzamelingen verwijderen,
maar opnieuw gewoon een stuk of 100,
en dan weer veronderstellen dat dat ongeveer representatief is.
Maar goed, dat is nog altijd 100 maal 100 oftewel 10.000 keer het model uitvoeren.
Dat is nog altijd veel rekenwerk.

Stel dat we een decompositie zouden hebben van het model,
met andere woorden:
voor elk van deze componenten hebben we een kleiner model
dat we kunnen berekenen.
Dan zou het al veel makkelijker zijn:
we moeten gewoon de uitvoer van elke component 1 keer berekenen in plaats van 100.
Het zijn er wel nog altijd veel: voor 3 variabelen zoals op de figuur zijn het er misschien maar 8,
maar voor 7 variabelen zijn het er nog steeds 128.
Maar daar kunnen we een beetje in snoeien.
In de praktijk is het namelijk zo dat voor veel datasets en modellen,
interacties tussen grote groepen variabelen eigenlijk redelijk zeldzaam zijn.
Meestal kan een model gesplitst worden in directe effecten en interacties tussen een paar variabelen.
Dat betekent dus, als we bijvoorbeeld 7 variabelen hebben,
we misschien al een goede benadering kunnen krijgen als we bijvoorbeeld alle
interacties tussen 3 variabelen of minder in rekening brengen.

Als we dus een decompositie van het model zouden hebben,
dan zouden we SHAP veel sneller kunnen berekenen.
Natuurlijk, ons model is te ingewikkeld om manueel op te splitsen
zoals we daarnet gedaan hebben met de functie.
Was er maar een manier om,
uit een grote hoeveelheid data bijvoorbeeld,
vanzelf een model als het ware te laten "leren"
voorspellen wat de uitvoer zou zijn van zo'n component...

### PDD-SHAP part 3 (1m30s)
Dat is dus natuurlijk precies wat mijn algoritme, PDD-SHAP, doet.
Kort samengevat, traint dit algoritme voor iedere component
een klein model,
zodat we uiteindelijk een verzameling kleine modellen hebben
die elk de waarde van een interactie-effect voorspellen.
Eens dat gebeurd is, kunnen we SHAP-waarden berekenen door gewoon
de uitvoer te berekenen van de kleine modellen,
en die uitvoer op de juiste manier te verdelen onder de variabelen.

Laat ons kijken naar het resultaat.
Wat je kan zien op deze grafiek is de accuraatheid van de SHAP attributies
zoals voorspeld door mijn algoritme.
Als die waarde 1 is, is de benadering perfect.
Iedere curve komt overeen met een bepaalde dataset, dus een bepaald model.
Op de X-as staat de grootte van de interacties die we modelleren,
hier bijvoorbeeld modelleren we alle directe effecten,
en interacties tussen 2 of 3 variabelen.
Voor de meeste datasets is de benadering redelijk goed vanaf 2 of 3,
voor andere datasets hebben we toch interacties tussen 4 variabelen nodig,
en misschien zelfs tussen 5 variabelen ook.

Op deze grafiek zien we dan de snelheid van het algoritme.
De Y-as is nu het aantal seconden dat nodig is om 1000 SHAP-waarden te berekenen.
Dit is een logaritmische as: hier zien we dus bijvoorbeeld dat mijn algoritme ongeveer 1000 keer
sneller is dan de klassieke algoritmes voor SHAP-waarden.
We krijgen dus een nieuwe afweging:
als we kunnen leven met een benadering,
en we hebben genoeg met relatief kleine interacties,
dan kunnen we dit algoritme gebruiken om heel veel rekenwerk te besparen.

## Conclusie (0m45s)
Om af te ronden: in mijn doctoraat heb ik 3 dingen gedaan.
Ten eerste heb ik experimenteel aangetoond 
dat er niet zoiets bestaat als een universele maat van "correctheid" van attributies,
en dat we eigenlijk moeten proberen om de verschillen tussen bestaande methoden beter te begrijpen,
in plaats van gewoon te zoeken naar "de beste."

Daarna heb ik een theoretisch kader ontworpen waarmee we een grote verzameling bestaande methoden
onder dezelfde noemer kunnen plaatsen, en dus ook makkelijker kunnen vergelijken,
en heb ik een link aangetoond tussen attributie en additieve decompositie van functies.

En ten slotte heb ik mijn theoretisch kader gebruikt
om een nieuw benaderingsalgoritme te ontwerpen voor een bestaande attributiemethode,
zodat die attributies veel sneller berekend kunnen worden.

Dan wil ik u alleen nog bedanken voor uw aandacht.

## Notes
- [ ] Use arrays in MathTex to make animations better
- [ ] Use DecimalNumber for learning animations
- [ ] Mention toeslagenaffaire in very beginning as a use case of XAI necessity
