import { Component, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { Observable } from 'rxjs';
import { startWith, map } from 'rxjs/operators';

import { AutocompleteService } from './autocomplete.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  myControl = new FormControl();
  filteredOptions: Observable<{ value: string, label: string }[]>;
  suggestions: string[] = [];

  constructor(
    private autocompleteService: AutocompleteService
  ) {}

  ngOnInit() {
    this.myControl.valueChanges
      .pipe(
        startWith('')
      ).subscribe(inputString => {
        this.filteredOptions = this.autocompleteService.getAutocompleteOptions(inputString).pipe(
          map(suggestions => suggestions.map(s => ({ value: `${inputString} ${s}`.replace('  ', ' '), label: s })))
        );
      });
  }
}
