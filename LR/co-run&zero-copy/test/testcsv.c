#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(void){

  const int taille = 546; // nombre de ligne du fichier
  
  FILE *ptr_file; // pointeur de fichier
  char buf[taille]; // tableau contenant la ligne courante
  double bufx[taille]; // tableau contenant le x
  double bufy[taille]; // tableau contenant le y
  char *token; // token utiliser pour strtok
  int c=0; // compteur de ligne (utile pour le remplissage des bufx et bufy
  char str_c; // jsais plus 
  char delim[] = "\t"; // delimiteur pour strtok
  ptr_file = fopen("newhouse.txt", "r"); // fichier a ouvrir
  

  /* Gestion des erreurs d'ouverture */
  if(!ptr_file) {
    return 1;
  }
  while (fgets(buf, taille, ptr_file)!=NULL){
    printf("%s",buf);     
    token = strtok(buf, delim); // on coupe la chaine de char
    bufx[c] = atof(token); // on la converti, et on la range dans le tab x
    printf("%s", token);
    token = strtok(NULL, delim); // on recupere ce qu'il reste
    bufy[c] = atof(token); // on la converti et on range dans tab y
    printf("\n");
    printf("%s", token);
    c = c + 1; // incrementation du compteur
    printf("---------------\n\n");
  }

  /* Fermeture du fichier */
  fclose(ptr_file);

  /* affichage du tableau de malade mental couz1 */
  for (int i=0; i<c; i++) {
    printf (" %f || %f",bufx[i], bufy[i]);
    printf("\n");
  }
  
  
  
  // static void invert_file() {
//   FILE *ptr_file = fopen(DATASET_FILENAME, "r");
//   FILE *ptr_write = fopen("assets/invert.txt", "w+");
//   if (ptr_write == NULL) {
//     perror("Failed to load target file");
//     exit(1);
//   }

//   char *token1, *token2;
//   char buf[1024];

//   for (int i = 0; i < DATASET_SIZE && fgets(buf, 1024, ptr_file) != NULL; i++) {
//     token1 = strtok(buf, "\t");
//     token2 = strtok(NULL, "\t");
//     token2[strlen(token2) - 2] = '\0';
//     fprintf(ptr_write, "%s\t%s\n", token2, token1);
//   }
// }

  
}
