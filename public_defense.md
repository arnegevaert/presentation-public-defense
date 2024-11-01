# Public defense

See also: 3blue1brown manim tutorial!
See also: manim slides! https://github.com/jeertmans/manim-slides

## Preamble: 1m00s
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

## Introductie: 9m00s

### Intro part 1 (1m45s)
Om te beginnen, moeten we even terug naar iedereen zijn/haar favoriete plek en meest dierbare herinnering:
de wiskundeles in het derde middelbaar.
Ik voel de spanning al een beetje toenemen in de zaal, maar er is geen reden om bang te zijn.
Ik heb namelijk gemerkt dat de meeste mensen die een trauma hebben aan wiskunde
eigenlijk vooral een trauma hebben aan de wiskunde*toets.*
Vandaag is er geen toets, dus ook geen reden om bang te zijn.

We beginnen met het concept *functie.*
Een echte informaticus denkt in termen van in- en output,
en we kunnen een functie ook zo bekijken:
een soort "machine" die een invoer neemt en een uitvoer teruggeeft.

Onze "machine" heeft ook een soort "programmacode" of "instructies"
die duidelijk maken wat de machine met zijn invoer moet doen.
We schrijven die programmacode in een vergelijking:
aan de linkerkant hebben we f van x, dus f is onze machine en x is de invoer,
en aan de rechterkant schrijven we wat we willen dat f doet met x.
In dit geval gaat f het getal x vermenigvuldigen met 2 en er dan 1 van aftrekken.

We kunnen dit eens testen: als we het getal 3 meegeven met f,
dan krijgen we inderdaad 2 maal 3 plus 1, en dat is 7.
We kunnen hetzelfde doen met -1, en dan krijgen we 2 maal -1 plus 1, en dat is -3.

We kunnen daar een grafiekje van maken.
Hier zien we onze twee punten terug, die komen elk overeen met een invoer (links-rechts)
en de bijhorende uitvoer (de hoogte).
Als we dit herhalen voor alle getallen, krijgen we deze lijn te zien.

### Intro part 2 (2m40s)
Laten we nog eens kijken naar onze formule.
Er zijn twee getallen die onze formule eigenlijk vastleggen: die 2 en die -1.
Als we een beetje prutsen aan die 2, dan zien we dat de richting van onze lijn verandert.
Als we prutsen aan die -1, dan gaat onze lijn naar boven en beneden.
Die 2 en die -1, we noemen die ook *parameters.*

Machine learning, is eigenlijk niets anders dan de computer 
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
Laten we dat eens doen, en voila, de lijn gaat mooi door de data.
We zien ook dat de foutfunctie nu veel kleiner is dan daarnet.

Nu denk je misschien, maar allez, we hadden die lijn nu toch ook wel met het blote oog kunnen tekenen.
En dat klopt, maar dat gaat niet altijd lukken.
Stel bijvoorbeeld dat we niet alleen de temperatuur bijhouden,
maar ook hoeveel mensen er thuis zijn,
en hoe hard de zon schijnt, want er liggen zonnepanelen op ons dak.
Nu moeten we dus 3 getallen meegeven als invoer aan het model in plaats van 1.
Die 3 getallen noemen we ook de *invoerveranderlijken.*
Met 3 invoerveranderlijken kunnen we niet meer een grafiekje tekenen zoals daarnet,
maar voor de computer is dit geen probleem.

### Intro part 3 (3m00s)
De functie waarvan we de parameters door de computer laten zoeken,
noemen we ook een *model.*
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
maar als we een parameter toevoegen aan onze functie,
dan lukt het wel: nu kan onze functie rechte lijnen en dit soort kromme lijnen tekenen.
Maar: nu moeten we wel 3 parameters laten kiezen door de computer,
en onze formule wordt ook een beetje moeilijker om te interpreteren.

Stel nu dat onze data er zo uitziet. *(complexe data, NN example)*
We kunnen opnieuw een functie maken en de parameters laten kiezen door de computer,
en dat ziet er dan zo uit.
Laten we nu eens kijken naar de programmacode van onze functie.
Oei, er is precies iets verkeerd met mijn slides...
Ah nee, ok.
Welke functie is dit? 
Dit is een *neuraal netwerk,* een speciaal soort functie dat we veel gebruiken in machine learning
en waarvan we kunnen kiezen hoeveel parameters het heeft.
Hoe meer parameters, hoe complexer de vormen die de functie kan tekenen.
Dit neuraal netwerk heeft 100 parameters, en we zien dus dat het redelijk moeilijk wordt
om de formule echt te verstaan.

Hoeveel parameters zitten er dan in een "echt" machine learning-model, zoals ChatGPT?
Toen ChatGPT 2 jaar geleden uitkwam, gebruikte dat het GPT3.5-model.
Dat model heeft *175* parameters.
Sorry, 175 *miljard* parameters.
Hoeveel parameters er in de nieuwste versie zitten, willen de makers ons niet vertellen,
maar het wordt geschat rond de 100 *biljoen*, dus nog eens ongeveer 1000 keer zoveel.

### Intro part 4 (1m20s)
Als we alle 100 biljoen parameters van ChatGPT op standaard A4 printerpapier zouden zetten,
dan zou de stapel papier ongeveer 25000km hoog zijn,
oftewel ongeveer vijf keer de afstand van deze zaal tot in Tashkent,
de hoofdstad van Oezbekistan.
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

## Benchmark

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

Nu vraag je je misschien af, hoe kan ik zo'n foto als invoer geven aan een functie?
Simpel: iedere pixel in de foto is een getal dat weergeeft hoe helder die pixel is.
In een kleurenfoto is iedere pixel 3 getallen: de hoeveelheid rood, groen en blauw in de pixel.
Dus als we zoals hier een klein zwart-wit fotootje hebben van 28 op 28 pixels,
dan heeft onze functie 784 invoerveranderlijken nodig.

Attributies zijn heel handig in dit geval.
Waarom? Omdat iedere invoerveranderlijke, dus iedere pixel, een score krijgt.
We kunnen dus iedere pixel een kleurtje geven die die score weergeeft,
en dan krijgen we dit soort visualizatie.
Dit zegt ons, voor een bepaalde foto, welke pixels het *belangrijkst* waren.
Merk op: nu hebben we een score voor 1 specifieke foto.
We noemen dat ook *lokale* attributies.
Daarnet, in ons model om energie te voorspellen,
beschreven de scores de invloed van de veranderlijken in het algemeen,
los van een specifieke meting.
Dat noemen we dan *globale* attributies.

Wat is nu het nut van zulke lokale attributies?
Wel, we kunnen ze bijvoorbeeld gebruiken om te kijken of het model wel geleerd heeft
wat we willen dat het leert.
Een kleine, waargebeurde anecdote.
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

### Benchmark part 2
Er zijn heel veel mensen die onderzoek doen naar goeie manieren om zo'n attributies te berekenen,
en dus zijn er heel veel manieren ontwikkeld om dat te doen.




- Er bestaan enorm veel manieren om een "heatmap" te maken voor een bepaalde predictie
  - Allemaal gebaseerd op speciale wiskundige technieken waar we nu niet op hoeven ingaan
  - Allemaal ontworpen om te tonen *welke regio's in de foto het belangrijkst waren voor die ene predictie.*
    - Waarom is dat nuttig? Zo kunnen we zien of het model echt geleerd heeft wat we willen dat het leert.
    - Husky voorbeeld/meetlat voorbeeld.
  - In veel gevallen genereren ze andere heatmaps.
    - Probleem: welke is nu de juiste?
- Simpele metriek: als we de 10% "belangrijkste" pixels vervangen door een grijs vierkant, verandert de predictie dan?
- Er bestaan enorm veel andere manieren om de kwaliteit van een heatmap te meten
  - Allemaal gebaseerd op speciale wiskundige technieken waar we nu niet op hoeven ingaan
  - Allemaal ontworpen om te meten *hoe "correct" de heatmap is:* zijn de "belangrijkste" regio's ook echt belangrijk?
- Simpele oefening:
  - We verzamelen een aantal datasets met afbeeldingen
  - We verzamelen een aantal methodes voor het genereren van heatmaps
  - We verzamelen een aantal kwaliteitsmetrieken voor heatmaps
  - We meten de kwaliteit van de verschillende heatmaps met de verschillende metrieken
- Resultaat:
  - "Beste" methode hangt af van de dataset...
  - ... maar ook van de metriek!
    - Conclusie: *er is niet 1 enkele maat van "correctheid".* 
      De metrieken meten toch nog onderling verschillende eigenschappen van de heatmaps.

## Removal-based attribution methods
- Stel: tabular model dat risico op diabetes moet bepalen obv lengte, leeftijd, geslacht, bloeddruk.
- We willen schatten hoe belangrijk iedere veranderlijke is.
  - De eerste vraag die we moeten beantwoorden is: voor wat?
    - bv: hoe gevoelig is het model aan iedere veranderlijke?
    - of: hoe nuttig is iedere veranderlijke voor de accuraatheid van ons model?
    - of: voor een bepaalde patient en predictie, hoe belangrijk is iedere veranderlijke voor die patient?
  - **Ingredient 1** van een verklaring: het **doelwit.**
- Stel nu: we hebben 1 patient met een predictie, en we willen weten hoe belangrijk de variabelen zijn voor die predictie.
  - Ons "doelwit" is gekozen.
  - Stel dat we willen weten hoe belangrijk "leeftijd" was.
    - Het liefst zouden we vragen aan het model: *Als we leeftijd niet kenden, wat zou dan je predictie zijn?*
      - Probleem: we moeten een waarde meegeven aan de functie.
      - Kunnen we een "lege waarde" meegeven? Op 0 zetten?
        - Probleem: dit is niet een "lege waarde", dit is "een pasgeboren baby"... Er zitten wss geen mensen met leeftijd 0 in de data
      - We kunnen alle waarden voor "leeftijd" eens invullen en kijken naar de gemiddelde uitvoer?
        - Probleem: dan krijgen we nog steeds rare combinaties, bvb 12-jarig kind met cholesterol van een 40-jarige *(todo beter voorbeeld?)*
      - Er zijn enorm veel manieren om een veranderlijke te "verwijderen", elk met voor- en nadelen.
      - **Ingredient 2** van een verklaring: de manier om variabelen te **verwijderen.**
- Stel nu: ons model is heel simpel, het voorspelt een verhoogde kans als x1 > a OF x2 > b.
  - Wat gaat onze simpele methode doen? Geen enkele variabele zal belangrijk zijn!
  - We mogen niet alleen kijken naar iedere variabele apart, maar we moeten ook kijken naar verzamelingen.
    - Dit is een voorbeeld van *interactie:* het effect van de twee variabelen samen is anders dan de som van het effect van de aparte veranderlijken.
  - Voor 4 variabelen hebben we 16 mogelijke deelverzamelingen
  - Hoe brengen we dit terug naar 1 score per veranderlijke?
  - **Ingredient 3** van een verklaring: de manier om de output van groepen van veranderlijken **terug samen te voegen** in 1 score per veranderlijke.
- We kunnen dit formeler maken.
  - Doelwit: Phi(f)
  - Verwijderen: P_S(f)
  - Aggregeren: alpha
- **Contributie:** we kunnen heel veel bestaande technieken onder deze noemer brengen
  - Dit maakt verschillende methodes plots makkelijker vergelijkbaar
    - Bvb methode A en B verschillen enkel in hoe ze veranderlijken verwijderen

## Functional decomposition
- Gegeven simpele functie (polynomial) 2x1 + 4x2 - 3x1x2 + x3 + 5
  - We kunnen deze functie schrijven als een som van functies die elk afhangen van een deelverzameling variabelen
    - De componenten met 1 variabele zijn "directe effecten", de componenten met meerdere zijn "interacties"
    - Voorbeeld van onze functie op een bepaald punt: we kunnen ieder component zien als een "bijdrage" van een (groep) veranderlijke(n)
  - Dit is een *additieve decompositie* (decompositie: we splitsen de functie in deeltjes, additief: het is een som)
    - De additieve decompositie is niet uniek: we kunnen termen naar boven verplaatsen
    - Maar: we mogen termen niet "naar beneden" verplaatsen
- Wat blijkt: iedere geldige additieve decompositie komt precies overeen met een manier van verwijderen P_S(f)
  - Dus: iedere attributiemethode uit ons framework kunnen we verbinden met een additieve decompositie
  - We kunnen "manier van verwijderen" vervangen door "manier van functie ontbinden"

## PDD-SHAP
- Wat is daar nu het praktisch nut van?
=> SHAP is een populaire methode: verwijdert alle mogelijke deelverzamelingen en aggregeert op een ingewikkelde manier.
Gebaseerd op speltheorie, theorie uit 1970(?) die oorspronkelijk ontwikkeld was om een bepaalde uitkomst van samenwerking eerlijk te verdelen onder spelers.

Maar: via framework kunnen we het samenvatten in 3 keuzes:
- doelwit = functie zelf, de scores zeggen wat de invloed is van iedere veranderlijke op de uitvoer van het model in x
- verwijderen = gemiddelde output over marginal distribution
- aggregeren = iedere interactie wordt gelijk verdeeld onder de leden

Computationele kost: aantal deelverzamelingen * aantal datapunten, super duur.

Als we de decompositie zouden hebben, kunnen we het gewoon berekenen als gewogen som van componenten.

Dit is het idee van PDD-SHAP: maak een nieuw model voor iedere component!

Te veel componenten? => factor sparsity

Results: R2, speed.


## Conclusie
- Benchmark => geen universele maat van "kwaliteit"
- Decompositie => we kunnen verschillende methodes vergelijken met elkaar door ze onder hetzelfde raamwerk te brengen
- PDD-SHAP => we kunnen dit raamwerk gebruiken om bestaande verklaringen sneller te berekenen







