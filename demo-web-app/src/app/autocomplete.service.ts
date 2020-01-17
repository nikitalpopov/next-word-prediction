import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

interface AutocompleteSuggestions {
  suggestions: Array<string>;
}

@Injectable()
export class AutocompleteService {
  constructor(
    private http: HttpClient
  ) { }

  getAutocompleteOptions(searchText: string): Observable<Array<string>> {
    return this.http.get<AutocompleteSuggestions>(`http://localhost:5000?input=${searchText.toLowerCase()}`).pipe(
      map(value => (value.suggestions))
    );
  }
}
